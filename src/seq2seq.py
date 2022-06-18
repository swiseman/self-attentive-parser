import os
import torch
import torch.nn as nn
import nltk

from transformers import AutoTokenizer, AutoModel, AutoConfig, T5ForConditionalGeneration
#from transformers.utils import PaddingStrategy
#from transformers.tokenization_utils import TruncationStrategy
from torch.nn.utils.rnn import pad_sequence

from benepar import retokenization, decode_chart, nkutil, subbatching

class Seq2seqParser(nn.Module):
    def __init__(self, label_vocab, hparams, pretrained_model_path=None):
        super().__init__()
        self.label_vocab = label_vocab
        self.retokenizer = self.retokenizer = retokenization.Retokenizer(
            hparams.pretrained_model, retain_start_stop=True)
        """
        if "bert" in hparams.pretrained_model:
            assert False
            tokenizer = self.retokenizer.tokenizer
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                hparams.pretrained_model, hparams.pretrained_model)
            model.config.decoder_start_token_id = tokenizer.cls_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
            model.config.vocab_size = model.config.decoder.vocab_size
            self.pretrained_model = model
        else:
        """
        #self.pretrained_model = AutoModel.from_config(
        #    AutoConfig.from_pretrained(hparams.pretrained_model))
        self.pretrained_model = T5ForConditionalGeneration.from_pretrained(hparams.pretrained_model)
        # self.chart_decoder = decode_chart.ChartDecoder(
        #     label_vocab=self.label_vocab, force_root_constituent=hparams.force_root_constituent)
        self.closing_label, self.dummy_word = hparams.closing_label, hparams.dummy_word
        if hparams.add_label_tokens:
            voc = self.retokenizer.tokenizer.vocab
            self.retokenizer.tokenizer.add_tokens(
                [k for k in self.label_vocab.keys() if k not in voc])
            self.pretrained_model.resize_token_embeddings(len(self.retokenizer.tokenizer))

    @classmethod
    def from_trained(cls, model_path):
        if os.path.isdir(model_path):
            # Multi-file format used when exporting models for release.
            # Unlike the checkpoints saved during training, these files include
            # all tokenizer parameters and a copy of the pre-trained model
            # config (rather than downloading these on-demand).
            config = AutoConfig.from_pretrained(model_path).benepar
            state_dict = torch.load(
                os.path.join(model_path, "benepar_model.bin"), map_location="cpu"
            )
            config["pretrained_model_path"] = model_path
        else:
            # Single-file format used for saving checkpoints during training.
            data = torch.load(model_path, map_location="cpu")
            config = data["config"]
            state_dict = data["state_dict"]

        hparams = config["hparams"]

        if "force_root_constituent" not in hparams:
            hparams["force_root_constituent"] = True

        config["hparams"] = nkutil.HParams(**hparams)
        parser = cls(**config)
        parser.load_state_dict(state_dict)
        return parser

    def encode(self, example):
        encoded = self.retokenizer(example.words, example.space_after)
        if example.tree is not None:
            tree = decode_chart.collapse_unary_strip_pos(example.tree)
            lintree = decode_chart.my_pformat_flat(
                tree, closing_label=self.closing_label, dummy_word=self.dummy_word)
            treetokes = self.retokenizer.tokenizer(lintree.split(), is_split_into_words=True)
            #encoded['decoder_input_ids'] = treetokes['input_ids']
            #encoded['decoder_attention_mask'] = treetokes['attention_mask']
            encoded["labels"] = treetokes['input_ids']
            #self.chart_decoder.chart_from_tree(example.tree)
        return encoded

    def pad_encoded(self, encoded_batch):
        #batch = self.retokenizer.pad(
        #    [{k: v for k, v in example.items()} for example in encoded_batch],
        #    return_tensors="pt")
        #batch = self.retokenizer.tokenizer.pad(
        #    [{k: v for k, v in example.items() if k != 'words_from_tokens'}
        #     for example in encoded_batch],
        #    return_tensors="pt")
        batch = {k: pad_sequence(
            [torch.LongTensor(example[k]) for example in encoded_batch], batch_first=True,
            padding_value=self.retokenizer.tokenizer.pad_token_id)
                 for k in encoded_batch[0].keys() if k not in ['labels', 'words_from_tokens']}
        if 'labels' in encoded_batch[0]:
            # labels needs a different pad token
            batch['labels'] = pad_sequence(
                [torch.LongTensor(example['labels']) for example in encoded_batch], batch_first=True,
                padding_value=-100)
        return batch

    def _get_lens(self, encoded_batch):
        if self.pretrained_model is not None:
            return [len(encoded["input_ids"]) for encoded in encoded_batch]
        return [len(encoded["valid_token_mask"]) for encoded in encoded_batch]

    def encode_and_collate_subbatches(self, examples, subbatch_max_tokens):
        batch_size = len(examples)
        batch_num_tokens = sum(len(x.words) for x in examples)
        encoded = [self.encode(example) for example in examples]

        res = []
        for ids, subbatch_encoded in subbatching.split(
            encoded, costs=self._get_lens(encoded), max_cost=subbatch_max_tokens):
            subbatch = self.pad_encoded(subbatch_encoded)
            subbatch["batch_size"] = batch_size
            subbatch["batch_num_tokens"] = batch_num_tokens
            res.append((len(ids), subbatch))
        return res

    def forward(self, batch):
        #valid_token_mask = batch["valid_token_mask"].to(self.output_device)
        # for some reason we don't do device stuff in main
        device = self.pretrained_model.lm_head.weight.device
        dec_attn_mask = (batch['labels'] != self.retokenizer.tokenizer.pad_token_id).to(
            batch['attention_mask'].dtype).to(device)
        loss = self.pretrained_model(
            input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device),
            decoder_attention_mask=dec_attn_mask.to(device), labels=batch['labels'].to(device)).loss # averages?
        return loss

    def compute_loss(self, batch):
        return self(batch)

    def _parse_encoded(self, examples, encoded):
        import ipdb; ipdb.set_trace()
        device = self.pretrained_model.lm_head.weight.device
        with torch.no_grad():
            batch = self.pad_encoded(encoded)
            gens = self.pretrained_model.generate(batch['input_ids'].to(device))
            gens = self.retokenizer.tokenizer.decode(gens)
            if self.closing_label:
                # maybe try to turn them into trees
                for thing in self.label_vocab:
                    closep = thing + ")"
                    for g, gen in enumerate(gens):
                        gens[g] = gen.replace(closep, ")")
            for g, gen in enumerate(gens):
                gens[g] = nltk.tree.Tree.fromstring(gen)

        for i in range(len(encoded)):
            leaves = examples[i].pos()
            if self.dummy_word is not None:
                assert False
            yield gens[i]
            #yield self.decoder.tree_from_chart(charts_np[i], leaves=leaves)

    def parse(self, examples, subbatch_max_tokens=None):
        training = self.training
        self.eval()
        encoded = [self.encode(example) for example in examples]
        if subbatch_max_tokens is not None:
            res = subbatching.map(
                self._parse_encoded, examples, encoded, costs=self._get_lens(encoded),
                max_cost=subbatch_max_tokens)
        else:
            res = self._parse_encoded(examples, encoded)
            res = list(res)
        self.train(training)
        return res
