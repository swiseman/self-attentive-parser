import os
import torch
import torch.nn as nn
import nltk

from transformers import EncoderDecoderModel, AutoTokenizer, AutoModel, AutoConfig

from benepar import retokenization, decode_chart, nkutil, subbatching

class Seq2seqParser(nn.Module):
    def __init__(self, label_vocab, hparams, pretrained_model_path=None):
        super().__init__()
        self.label_vocab = label_vocab
        self.retokenizer = self.retokenizer = retokenization.Retokenizer(
            hparams.pretrained_model, retain_start_stop=True)
        if "bert" in hparams.pretrained_model:
            tokenizer = self.retokenizer.tokenizer
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                hparams.pretrained_model, hparams.pretrained_model)
            model.config.decoder_start_token_id = tokenizer.cls_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
            model.config.vocab_size = model.config.decoder.vocab_size
            self.pretrained_model = model
        else:
            self.pretrained_model = AutoModel.from_config(
                AutoConfig.from_pretrained(hparams.pretrained_model))
        # self.chart_decoder = decode_chart.ChartDecoder(
        #     label_vocab=self.label_vocab, force_root_constituent=hparams.force_root_constituent)
        self.closing_label, self.dummy_word = hparams.closing_label, hparams.dummy_word

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
            lintokes = self.retokenizer(lintree, example.space_after)
            encoded["labels"] = lintokes
            #self.chart_decoder.chart_from_tree(example.tree)
        return encoded

    def pad_encoded(self, encoded_batch):
        batch = self.retokenizer.pad(
            [{k: v for k, v in example.items()} for example in encoded_batch],
            return_tensors="pt")
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
        valid_token_mask = batch["valid_token_mask"].to(self.output_device)
        loss, logits = self.pretrained_model(*batch) # averages?

    def compute_loss(self, batch):
        loss, _ = self(batch)
        return loss

    def _parse_encoded(self, examples, encoded):
        with torch.no_grad():
            batch = self.pad_encoded(encoded)
            gens = self.pretrained_model.generate(batch['input_ids'])
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
                max_cost=subbatch_max_tokens, return_compressed=return_compressed,
                return_scores=return_scores, return_amax=return_amax)
        else:
            res = self._parse_encoded(examples, encoded, return_compressed=return_compressed,
                                      return_scores=return_scores)
            res = list(res)
        self.train(training)
        return res
