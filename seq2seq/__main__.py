# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import logging
import distutils.util
import os


from seq2seq.train import train_loop
from seq2seq.utils import print_config
from seq2seq.test import test

import torch


logging.basicConfig(
    format='%(asctime)-15s %(levelname)s %(name)s: %(message)s', level=logging.INFO,
    datefmt='%Y%m%d %H:%M:%S')
logger = logging.getLogger(__name__)


def main_loop(args):
    """
    Choose between train or test
    """
    use_cuda = torch.cuda.is_available()
    logger.info("CUDA: %s" % use_cuda)

    args = vars(args)
    print_config(args)
    mode = args['mode']

    if mode == "train":
        train_loop(**args)
    elif mode == "test":
        test(**args)
    else:
        raise ValueError("Unknown mode, choose train or test")


def main():
    print(os.getcwd())

    """
    Handle args
    """
    ap = argparse.ArgumentParser()

    ap.add_argument('--mode', choices=['train', 'test'], default='train',
                    help="train a model or test (predict)", required=True)

    ap.add_argument('--model_type', choices=['encdec'], default='encdec')

    ap.add_argument('--enc_type', choices=['rnn', 'birnn'], default='birnn',
                    help='encoder type; unidirectional=rnn, bidirectional=birnn')
    ap.add_argument('--dec_type', choices=['rnn', 'rnn_att'], default='rnn_att',
                    help='decoder type; rnn for simple enc-dec without '
                         'attention, rnn_att for dec with attention')

    ap.add_argument("--src", type=str, default="src",
                    help="source language postfix")
    ap.add_argument("--trg", type=str, default="trg",
                    help="target language postfix")

    ap.add_argument("--root", type=str, default="/path/to/data/dir",
                    help="path to where the data set is located")
    ap.add_argument("--train", type=str, default="train",
                    help="train filename (must be located in `root`)")
    ap.add_argument("--validation", type=str, nargs='+', default="",
                    help="validation filename(s) (must be located in `root`); "
                         "supports 1-2 arguments")
    ap.add_argument("--test", type=str, default=None,
                    help="test filename (must be located in `root`)")

    ap.add_argument("--workdir", type=str, default=".",
                    help="working directory (for output); "
                         "when launched with aisweeper, will use environment "
                         "variable to set this automatically")

    ap.add_argument("--emb_dim", "-e", type=int, default=64,
                    help="word emb dim")
    ap.add_argument("--dim", "-d", type=int, default=256,
                    help="rnn/hidden dim")
    ap.add_argument("--batch_size", "-b", type=int, default=32,
                    help="batch size")
    ap.add_argument("--n_enc_layers", "--ne", type=int, default=1,
                    help="number of encoder layers")
    ap.add_argument("--n_dec_layers", "--nd", type=int, default=1,
                    help="number of decoder layers")

    ap.add_argument('--rnn_type', choices=['lstm', 'gru', 'rnn'],
                    default='gru',
                    help="rnn cell type")
    ap.add_argument('--use_visdom', action='store_true',
                    help="plot attention and progress on visdom server;"
                         "launch the server with `python -m visdom.server`")

    ap.add_argument("--n_val_examples", "--nv", type=int, default=5,
                    help="number of validation examples to show")

    ap.add_argument("--min_freq", type=int, default=0,
                    help="only include a word in the vocabulary if it occurs "
                         "at least this many times in the train set")
    ap.add_argument("--tf_ratio", type=float, default=1.0,
                    help="teacher forcing ratio; 1.0 = always use teacher forcing, "
                         "0.5 = use teacher forcing in 50% of mini-batches")

    # optimization
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--weight_decay", type=float, default=1e-8)
    ap.add_argument("--learning_rate", "--lr", type=float, default=0.2)
    ap.add_argument("--learning_rate_decay", "--lrd", type=float, default=0.96)
    ap.add_argument('--optimizer', choices=['adam', 'sgd'], default='sgd', help="optimizer")
    ap.add_argument("--clip", type=float, default=5.0, help="clip gradient norm")
    ap.add_argument("--n_iters", type=int, default=-150,
                    help="How many iteration to train. "
                         "Set to -`X` for training `X` epochs.")

    # attention architecture (default condition on all these things)
    # for Bahdanau, all should be 1.
    # for Luong, set predict_from_emb to 0.
    ap.add_argument('--predict_from_emb', type=distutils.util.strtobool, default=1)
    ap.add_argument('--predict_from_ctx', type=distutils.util.strtobool, default=1)
    ap.add_argument('--predict_from_dec', type=distutils.util.strtobool, default=1)
    ap.add_argument('--dec_input_emb', type=distutils.util.strtobool, default=1)
    ap.add_argument('--dec_input_ctx', type=distutils.util.strtobool, default=1)

    # misc
    ap.add_argument("--save_every", type=int, default=-1,
                    help="-1 for saving every epoch")
    ap.add_argument("--print_every", type=int, default=50,
                    help="print info every this many iterations (updates)")
    ap.add_argument("--plot_every", type=int, default=-1,
                    help="-1 for plotting every epoch")
    ap.add_argument("--eval_every", type=int, default=-1,
                    help="-1 for evaluation every epoch")

    ap.add_argument('--resume', type=str, default="",
                    help="path to checkpoint (if you want to resume training)")

    ap.add_argument("--max_length", type=int, default=50,
                    help="max supported sent length")
    ap.add_argument("--max_length_train", type=int, default=50,
                    help="max supported sent length")
    ap.add_argument('--metric',
                    choices=['exact_match', 'accuracy', 'perplexity', 'bleu'],
                    default='bleu', help="metric for model selection")

    # for test only
    ap.add_argument("--ckpt", type=str, default="checkpoint.best.pt.tar",
                    help="checkpoint to load for testing")
    ap.add_argument('--src_vocab', type=str, default="src_vocab.pt.tar",
                    help="src vocab filename")
    ap.add_argument('--trg_vocab', type=str, default="trg_vocab.pt.tar",
                    help="trg vocab filename")

    ap.add_argument('--save_heatmaps', type=distutils.util.strtobool, default=0)
    ap.add_argument('--external_bleu', type=distutils.util.strtobool, default=0)
    ap.add_argument('--debpe', type=distutils.util.strtobool, default=0)

    ap.add_argument('--unk_src', type=distutils.util.strtobool, default=1,
                    help="include UNK in source vocab")
    ap.add_argument('--unk_trg', type=distutils.util.strtobool, default=1,
                    help="include UNK in target vocab")
    ap.add_argument('--pass_hidden_state', type=distutils.util.strtobool,
                    default=1)

    # scan specific
    ap.add_argument('--scan_normalize', type=distutils.util.strtobool,
                    default=0,
                    help="normalize the output sequences so that B "
                         "after A becomes A and B")

    args = ap.parse_args()
    main_loop(args)


if __name__ == "__main__":
    main()
