import argparse
import functools
import itertools
import os.path
import time
import math
import torch

torch.backends.cuda.matmul.allow_tf32 = True

import numpy as np

from benepar import char_lstm
from benepar import decode_chart
from benepar import nkutil
from benepar import parse_chart
import evaluate
import learning_rates
import treebanks

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string


def make_hparams():
    return nkutil.HParams(
        # Data processing
        max_len_train=0,  # no length limit
        max_len_dev=0,  # no length limit
        # Optimization
        batch_size=32,
        learning_rate=0.00005,
        learning_rate_warmup_steps=160,
        clip_grad_norm=0.0,  # no clipping
        checks_per_epoch=4,
        step_decay_factor=0.5,
        step_decay_patience=5,
        max_consecutive_decays=3,  # establishes a termination criterion
        # CharLSTM
        use_chars_lstm=False,
        d_char_emb=64,
        char_lstm_input_dropout=0.2,
        # BERT and other pre-trained models
        use_pretrained=False,
        pretrained_model="bert-base-uncased",
        # Partitioned transformer encoder
        use_encoder=False,
        d_model=1024,
        num_layers=8,
        num_heads=8,
        d_kv=64,
        d_ff=2048,
        encoder_max_len=512,
        # Dropout
        morpho_emb_dropout=0.2,
        attention_dropout=0.2,
        relu_dropout=0.1,
        residual_dropout=0.2,
        # Output heads and losses
        force_root_constituent="auto",
        predict_tags=False,
        d_label_hidden=256,
        d_tag_hidden=256,
        tag_loss_scale=5.0,
        weight_decay=0.0001,
    )


def run_train(args, hparams):
    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)

    # Make sure that pytorch is actually being initialized randomly.
    # On my cluster I was getting highly correlated results from multiple
    # runs, but calling reset_parameters() changed that. A brief look at the
    # pytorch source code revealed that pytorch initializes its RNG by
    # calling std::random_device, which according to the C++ spec is allowed
    # to be deterministic.
    seed_from_numpy = np.random.randint(2147483648)
    print("Manual seed for pytorch:", seed_from_numpy)
    torch.manual_seed(seed_from_numpy)

    hparams.set_from_args(args)

    print("Loading training trees from {}...".format(args.train_path))
    train_treebank = treebanks.load_trees(
        args.train_path, args.train_path_text, args.text_processing
    )
    if hparams.max_len_train > 0:
        train_treebank = train_treebank.filter_by_length(hparams.max_len_train)
    print("Loaded {:,} training examples.".format(len(train_treebank)))

    print("Loading development trees from {}...".format(args.dev_path))
    dev_treebank = treebanks.load_trees(
        args.dev_path, args.dev_path_text, args.text_processing
    )
    if hparams.max_len_dev > 0:
        dev_treebank = dev_treebank.filter_by_length(hparams.max_len_dev)
    print("Loaded {:,} development examples.".format(len(dev_treebank)))

    print("Constructing vocabularies...")
    label_vocab = decode_chart.ChartDecoder.build_vocab(train_treebank.trees)
    if hparams.use_chars_lstm:
        char_vocab = char_lstm.RetokenizerForCharLSTM.build_vocab(train_treebank.sents)
    else:
        char_vocab = None

    tag_vocab = set()
    for tree in train_treebank.trees:
        for _, tag in tree.pos():
            tag_vocab.add(tag)
    tag_vocab = ["UNK"] + sorted(tag_vocab)
    tag_vocab = {label: i for i, label in enumerate(tag_vocab)}

    if hparams.force_root_constituent.lower() in ("true", "yes", "1"):
        hparams.force_root_constituent = True
    elif hparams.force_root_constituent.lower() in ("false", "no", "0"):
        hparams.force_root_constituent = False
    elif hparams.force_root_constituent.lower() == "auto":
        hparams.force_root_constituent = (
            decode_chart.ChartDecoder.infer_force_root_constituent(train_treebank.trees)
        )
        print("Set hparams.force_root_constituent to", hparams.force_root_constituent)

    hparams.mode, hparams.share_layers = args.mode, args.share_layers
    hparams.higher_order, hparams.ho_stuff = args.higher_order, args.ho_stuff
    hparams.stop_thresh = args.stop_thresh

    print("Hyperparameters:")
    hparams.print()

    print("Initializing model...")
    parser = parse_chart.ChartParser(
        tag_vocab=tag_vocab,
        label_vocab=label_vocab,
        char_vocab=char_vocab,
        hparams=hparams,
    )

    if args.parallelize:
        parser.parallelize()
    elif torch.cuda.is_available():
        parser.cuda()
    else:
        print("Not using CUDA!")

    print("Initializing optimizer...")
    #trainable_parameters = [
    #    param for param in parser.parameters() if param.requires_grad
    #]
    #optimizer = torch.optim.Adam(
    #    trainable_parameters, lr=hparams.learning_rate, betas=(0.9, 0.98), eps=1e-9
    #)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in parser.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": hparams.weight_decay,
        },
        {
            "params": [p for n, p in parser.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=hparams.learning_rate)

    if args.lrsched is None:
        scheduler = learning_rates.WarmupThenReduceLROnPlateau(
            optimizer,
            hparams.learning_rate_warmup_steps,
            mode="max",
            factor=hparams.step_decay_factor,
            patience=hparams.step_decay_patience * hparams.checks_per_epoch,
            verbose=True,)
    else:
        from transformers import get_constant_schedule_with_warmup
        scheduler = get_constant_schedule_with_warmup(optimizer, hparams.learning_rate_warmup_steps)

    """
    def lr_lambda(current_step):
        if current_step < hparams.learning_rate_warmup_steps:
            lr_step = (hparams.learning_rate - 1e-7)/hparams.learning_rate_warmup_steps
            return (1e-7 + current_step*lr_step)/hparams.learning_rate
        return hparams.learning_rate_warmup_steps**0.5 * current_step**-0.5
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    """

    #clippable_parameters = trainable_parameters
    #grad_clip_threshold = (
    #    np.inf if hparams.clip_grad_norm == 0 else hparams.clip_grad_norm
    #)

    print("Training...")
    total_processed = 0
    current_processed = 0
    check_every = len(train_treebank) / hparams.checks_per_epoch
    best_dev_fscore = -np.inf
    best_dev_model_path = None
    best_dev_processed = 0
    current_dev_fscore = None

    start_time = time.time()

    def check_dev():
        nonlocal best_dev_fscore
        nonlocal best_dev_model_path
        nonlocal best_dev_processed
        nonlocal current_dev_fscore

        dev_start_time = time.time()

        dev_predicted = parser.parse(
            dev_treebank.without_gold_annotations(),
            subbatch_max_tokens=args.subbatch_max_tokens,
        )
#        import ipdb; ipdb.set_trace()
        dev_fscore = evaluate.evalb(args.evalb_dir, dev_treebank.trees, dev_predicted)
        current_dev_fscore = dev_fscore.fscore

        print(
            "dev-fscore {} "
            "lr {:.5f} "
            "dev-elapsed {} "
            "total-elapsed {}".format(
                dev_fscore, scheduler.optimizer.param_groups[0]['lr'],
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )

        if args.nruns == 1 and dev_fscore.fscore > best_dev_fscore:
            if best_dev_model_path is not None:
                extensions = [".pt"]
                for ext in extensions:
                    path = best_dev_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_fscore = dev_fscore.fscore
            best_dev_model_path = "{}_dev={:.2f}".format(
                args.model_path_base, dev_fscore.fscore
            )
            best_dev_processed = total_processed
            print("Saving new best model to {}...".format(best_dev_model_path))
            torch.save(
                {
                    "config": parser.config,
                    "state_dict": parser.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                best_dev_model_path + ".pt",
            )

    data_loader = torch.utils.data.DataLoader(
        train_treebank,
        batch_size=hparams.batch_size,
        shuffle=True,
        collate_fn=functools.partial(
            parser.encode_and_collate_subbatches,
            subbatch_max_tokens=args.subbatch_max_tokens,
        ),
    )
    nsteps = 0
    best_sd = None
    for epoch in itertools.count(start=1):
        epoch_start_time = time.time()

        for batch_num, batch in enumerate(data_loader, start=1):
            optimizer.zero_grad()
            parser.train()
            #import ipdb; ipdb.set_trace()
            #(batch[0][1]['span_labels'] != -100).sum()

            totespans = 0.0
            batch_loss_value = 0.0
            for subbatch_size, subbatch in batch:
                if args.mode:
                    loss, nspans = parser.compute_loss2(subbatch)
                    totespans += nspans
                else:
                    loss = parser.compute_loss(subbatch)
                loss_value = float(loss.data.cpu().numpy())
                batch_loss_value += loss_value
                if loss_value > 0:
                    loss.backward()
                del loss
                total_processed += subbatch_size
                current_processed += subbatch_size

            if totespans > 0:
                for p in parser.parameters():
                    if p.grad is not None:
                        p.grad.detach().div_(totespans)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                #clippable_parameters, grad_clip_threshold
                parser.parameters(), hparams.clip_grad_norm
            )

            optimizer.step()
            nsteps += 1
            """
            print(
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "grad-norm {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    batch_num,
                    int(np.ceil(len(train_treebank) / hparams.batch_size)),
                    total_processed,
                    batch_loss_value,
                    grad_norm,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )
            """

            if current_processed >= check_every:
                current_processed -= check_every
                print("nsteps", nsteps)
                check_dev()
                if args.lrsched is None and ((math.isnan(current_dev_fscore) and best_dev_fscore >= 10)
                      or current_dev_fscore <= 0.5*best_dev_fscore):
                    print("rewinding to last good checkpt...")
                    parser.load_state_dict(best_sd)
                    scheduler._reduce_lr(scheduler.last_epoch+1)
                elif args.lrsched is not None: # no metrics stuff
                    scheduler.step()
                else:
                    scheduler.step(metrics=best_dev_fscore)
                if current_dev_fscore == best_dev_fscore: # we improved
                    best_sd = {k: v.clone() for k, v in parser.state_dict().items()}
            else:
                scheduler.step()

        if (total_processed - best_dev_processed) > (
            (hparams.step_decay_patience + 1)
            * hparams.max_consecutive_decays
            * len(train_treebank)
        ):
            print("Terminating due to lack of improvement in dev fscore.")
            break

        if args.max_epochs is not None and epoch >= args.max_epochs:
            break
        if args.max_steps is not None and nsteps >= args.max_steps:
            break

def run_test(args):
    print("Loading test trees from {}...".format(args.test_path))
    test_treebank = treebanks.load_trees(
        args.test_path, args.test_path_text, args.text_processing
    )
    print("Loaded {:,} test examples.".format(len(test_treebank)))

    if len(args.model_path) != 1:
        raise NotImplementedError(
            "Ensembling multiple parsers is not "
            "implemented in this version of the code."
        )

    model_path = args.model_path[0]
    print("Loading model from {}...".format(model_path))
    parser = parse_chart.ChartParser.from_trained(model_path)
    if args.no_predict_tags and parser.f_tag is not None:
        print("Removing part-of-speech tagging head...")
        parser.f_tag = None
    parser.stop_thresh = args.stop_thresh
    if args.parallelize:
        parser.parallelize()
    elif torch.cuda.is_available():
        parser.cuda()

    print("Parsing test sentences...")
    start_time = time.time()

    test_predicted = parser.parse(
        test_treebank.without_gold_annotations(),
        subbatch_max_tokens=args.subbatch_max_tokens,
    )

    if args.output_path == "-":
        for tree in test_predicted:
            print(tree.pformat(margin=1e100))
    elif args.output_path:
        with open(args.output_path, "w") as outfile:
            for tree in test_predicted:
                outfile.write("{}\n".format(tree.pformat(margin=1e100)))

    # The tree loader does some preprocessing to the trees (e.g. stripping TOP
    # symbols or SPMRL morphological features). We compare with the input file
    # directly to be extra careful about not corrupting the evaluation. We also
    # allow specifying a separate "raw" file for the gold trees: the inputs to
    # our parser have traces removed and may have predicted tags substituted,
    # and we may wish to compare against the raw gold trees to make sure we
    # haven't made a mistake. As far as we can tell all of these variations give
    # equivalent results.
    ref_gold_path = args.test_path
    if args.test_path_raw is not None:
        print("Comparing with raw trees from", args.test_path_raw)
        ref_gold_path = args.test_path_raw

    test_fscore = evaluate.evalb(
        args.evalb_dir, test_treebank.trees, test_predicted, ref_gold_path=ref_gold_path
    )

    print(
        "test-fscore {} "
        "test-elapsed {}".format(
            test_fscore,
            format_elapsed(start_time),
        )
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    hparams = make_hparams()
    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=lambda args: run_train(args, hparams))
    hparams.populate_arguments(subparser)
    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--train-path", default="data/wsj/train_02-21.LDC99T42")
    subparser.add_argument("--train-path-text", type=str)
    subparser.add_argument("--dev-path", default="data/wsj/dev_22.LDC99T42")
    subparser.add_argument("--dev-path-text", type=str)
    subparser.add_argument("--text-processing", default="default")
    subparser.add_argument("--subbatch-max-tokens", type=int, default=2000)
    subparser.add_argument("--parallelize", action="store_true")
    subparser.add_argument("--print-vocabs", action="store_true")
    subparser.add_argument("--mode", type=str, default=None, choices=["bce", "mlr"])
    subparser.add_argument("--higher-order", action="store_true", help="")
    subparser.add_argument("--share-layers", action="store_true", help="")
    subparser.add_argument("--ho-stuff", type=str, default=None, help="")
    subparser.add_argument("--lrsched", type=str, default=None, help="")
    subparser.add_argument("--stop-thresh", type=float, default=0.0)
    subparser.add_argument("--max-epochs", type=int, default=None)
    subparser.add_argument("--nruns", type=int, default=1)
    subparser.add_argument("--gseed", type=int, default=0)

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    subparser.add_argument("--model-path", nargs="+", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--test-path", default="data/wsj/test_23.LDC99T42")
    subparser.add_argument("--test-path-text", type=str)
    subparser.add_argument("--test-path-raw", type=str)
    subparser.add_argument("--text-processing", default="default")
    subparser.add_argument("--subbatch-max-tokens", type=int, default=500)
    subparser.add_argument("--parallelize", action="store_true")
    subparser.add_argument("--output-path", default="")
    subparser.add_argument("--no-predict-tags", action="store_true")
    subparser.add_argument("--stop-thresh", type=float, default=0.0)

    args = parser.parse_args()
    print(args)

    if args.nruns > 1:
        #args.max_epochs = 5
        args.max_steps = 5000
        args.checks_per_epoch = 1
        gen = torch.Generator()
        gen.manual_seed(args.gseed)
        grid = {
            # 'position_embedding_type': ['absolute', 'relative_key'],
            'clip_grad_norm': [0.1, 0.3],#, 10.0],
            'relu_dropout': [0.1],
            'share_layers': [False], #, True],
            'ho_stuff': [#"amlp+apoolall", "amlp+apool0",
                         #"afeats+apoolall", "afeats+apool0",
                         "mmlp+apoolall"], #"mmlp+apool0"],
            'learning_rate': [5e-5, 1e-4],
            'weight_decay': [1e-4, 1e-3],
            'learning_rate_warmup_steps': [160, 500, 1000],
            'batch_size': [64, 128],
            'lrsched': ["con+warm", None],
        }
        import itertools
        if args.gseed < 0:
            keys = list(grid.keys())
            lengths = [len(grid[key]) for key in keys]
            prods = itertools.product(*[range(n) for n in lengths])
            args.nruns = 10000

        for j in range(args.nruns):
            if args.gseed < 0:
                try:
                    prod = next(prods)
                    rdict = {keys[i]: grid[keys[i]][idx] for i, idx in enumerate(prod)}
                except StopIteration:
                    break
            else:
                rdict = {k: v[torch.randint(len(v), (1,), generator=gen).item()] for k, v in grid.items()}
            for k, v in rdict.items():
                args.__dict__[k] = v
            args.callback(args)
    else:
        args.callback(args)


if __name__ == "__main__":
    main()
