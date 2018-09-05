# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import matplotlib
matplotlib.use('Agg')

import time
import math
import os
import torch
import random
import logging
import visdom
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import imageio
from torchtext import data

logger = logging.getLogger(__name__)
use_cuda = torch.cuda.is_available()

SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

viz = visdom.Visdom()


def init_gru(cell, gain=1):
    cell.reset_parameters()

    # orthogonal initialization of recurrent weights
    for _, hh, _, _ in cell.all_weights:
        for i in range(0, hh.size(0), cell.hidden_size):
            torch.nn.init.orthogonal(hh[i:i + cell.hidden_size], gain=gain)


def get_fields(src=None, trg=None, unk_src=True, unk_trg=True,
               sos_src=False, sos_trg=False):
    """
    Create torchtext fields for data IO.
    """

    # load data
    src_field = data.Field(sequential=True, tokenize=str.split, batch_first=True, include_lengths=True,
                           eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, unk_token=UNK_TOKEN if unk_src else None,
                           init_token=SOS_TOKEN if sos_src else None)

    trg_field = data.Field(sequential=True, tokenize=str.split, batch_first=True, include_lengths=True,
                           eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, unk_token=UNK_TOKEN if unk_trg else None,
                           init_token=SOS_TOKEN if sos_trg else None)

    exts = ['.' + src, '.' + trg]
    fields = [('src', src_field), ('trg', trg_field)]

    return fields, exts


def print_config(args):
    """
    Prints complete configuration.
    """
    print("Configuration:")
    for k, v in sorted(args.items()):
        print("%s : %s" % (k, v))


def print_parameter_info(model):
    """
    Prints all trainable parameters (name and shape) as well as total count.
    """
    print("Model parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("%s : %s" % (name, [d for d in param.size()]))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    param_count = sum([np.prod(p.size()) for p in model_parameters])
    print("Total parameter count: %d" % param_count)


def as_minutes(s):
    """
    Returns time in minutes
    """
    m = math.floor(s / 60)
    s -= m * 60
    return '%02d:%02d' % (m, s)


def time_since(since, percent):
    """
    Returns how much time is left until completion, given a start time
    `since` and a percentage `percent` (iter/total_iters)
    """
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (-%s)' % (as_minutes(s), as_minutes(rs))


def save_checkpoint(state, path):
    """
    Saves model to disk.
    """
    torch.save(state, path)


def get_state_dict(iter, model_type, model_state_dict, optimizer_state_dict, stats):
    """
    Make a state dict for saving as a checkpoint
    """
    state = {
        'iter':            iter,
        'model_type':      model_type,
        'state_dict':      model_state_dict,
        'opt':             optimizer_state_dict,
        'stats': stats
    }
    return state


class Example(object):
    def __init__(self, d):
        self.__dict__.update(d)


def postprocess_examples(batch=None, fields=None, predictions=None,
                         attention_scores=None):

    examples = []

    src_var, src_lengths = batch.src
    trg_var, trg_lengths = batch.trg

    src_vocab = fields[0][-1].vocab
    trg_vocab = fields[1][-1].vocab

    for rid in range(len(batch.src[1])):

        src_seq = [src_vocab.itos[i] for i in src_var[rid].data.cpu().numpy()][:src_lengths[rid]]
        trg_seq = [trg_vocab.itos[i] for i in trg_var[rid].data.cpu().numpy()][:trg_lengths[rid]] if trg_var is not None else None

        # words
        if predictions is not None:
            prediction = []
            for token_id in predictions[rid].cpu().numpy():
                prediction.append(trg_vocab.itos[token_id])
                if token_id == trg_vocab.stoi[EOS_TOKEN]:
                    break
        else:
            prediction = None

        attention = attention_scores[rid, :len(prediction), :len(src_seq)].cpu().numpy() if attention_scores is not None else None

        example = Example(dict(src=src_seq, trg=trg_seq,
                               prediction=prediction,
                               attention_scores=attention))
        examples.append(example)

    return examples


def get_random_examples(batch, fields, predictions=None,
                        attention_scores=None, n=3, seed=None,):
    """
    Return n random examples from batch (without duplicates).

    Args:
        batch:
        fields:
        predictions:
        attention_scores:
        n:
        seed:

    Returns:

    """
    examples = postprocess_examples(batch, fields, predictions=predictions,
                                    attention_scores=attention_scores)
    if seed is not None:
        random.seed(seed)
    random.shuffle(examples)
    return examples[:n]


# def print_examples(src_var, trg_var, src_lengths, trg_lengths, src_vocab, trg_vocab, predictions=None, n=2):
def print_examples(examples, n=3, msg="Example", start="\n", end="\n"):
        """
        Print a few examples.

        Args:
            examples:
            n:
            msg:
            start:
            end:

        Returns:

        """
        if examples is None:
            return

        print(start, end="")
        for i, example in enumerate(examples[:n], 1):
            print("%s %d:" % (msg, i))
            print("  src  : %s" % " ".join(example.src))
            print("  tgt  : %s" % " ".join(example.trg))

            if example.prediction is not None:
                print("  pred.: %s" % " ".join(example.prediction))

            print(end, end="")


def plot_examples(examples, n=5, iteration=0, plot_file_fmt='iter%06d_example%d', workdir='.', use_visdom=False,
                  save_to_disk=True):
    """
    Plot the first n examples of a list of examples.

    Args:
        examples:
        n:
        iteration:
        plot_file_fmt:
        workdir:

    Returns:

    """
    for i, example in enumerate(examples[:n], 1):
        if example.attention_scores is not None:

            title = plot_file_fmt % (iteration, i)

            # visdom plot
            if use_visdom:
                plot_visdom_heatmap(example, title=title)

            if save_to_disk:
                # plot to disk (png)
                path = os.path.join(workdir, plot_file_fmt % (iteration, i) + ".png")
                plot_heatmap(example, path, title=title)

                # plot to disk (pdf)
                path = os.path.join(workdir, plot_file_fmt % (iteration, i) + ".pdf")
                plot_heatmap(example, path, title=title)


def plot_visdom_heatmap(example, title=""):
    """ Plot attention heatmap with Visdom. """

    attention = example.attention_scores

    # we need a workaround since plotly doesn't like non-unique names
    # so we add the index to make them unique
    columnnames = ['%d %s' % (i, w) for i, w in enumerate(example.src, 1)]
    rownames = ['%d %s' % (i, w) for i, w in enumerate(example.prediction, 1)]

    opts = dict(
        columnnames=columnnames,
        rownames=rownames,
        colormap='Viridis',
        xmin=0.,
        xmax=1.,
        font=dict(size=8),
        fillarea=True,
        title=title,
    )
    viz.heatmap(X=attention, opts=opts)


def plot_visdom_heatmap_simple(heatmap, title="", columnnames=None, rownames=None, colormap='Viridis'):
    """ Plot general heatmap with Visdom. """

    opts = dict(
        columnnames=columnnames,
        rownames=rownames,
        colormap=colormap,
        font=dict(size=8),
        fillarea=True,
        title=title,
    )
    viz.heatmap(X=heatmap, opts=opts)


def plot_heatmap(example, path='plot.png', title=""):
    """Plot heatmap with matpotlib. """

    font = {'size': 7}
    matplotlib.rc('font', **font)

    scores = example.attention_scores
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(scores, cmap='viridis', vmin=0., vmax=1.)

    # flip the scores; y-labels start at the top with pyplot
    # in plotly they would start at the bottom
    scores = scores[::-1]

    # set tick labels to the words
    ax.set_xticklabels(example.src, minor=False, rotation=90)
    ax.set_yticklabels(example.prediction, minor=False, rotation=0)
    # plt.axis('equal')

    ax.set_aspect(1)
    ax.set_title(title, y=1.02)

    # put the major ticks at the middle of each cell
    # and the x-ticks on top
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(scores.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(scores.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()

    # plt.colorbar(heatmap)
    if path.endswith('.pdf'):
        pp = PdfPages(path)
        pp.savefig()
        pp.close()
    else:
        plt.savefig(path, dpi=300)
        plt.close('all')


def build_model(model_type=None, **kwargs):
    """ General helper function used to construct any model given hyper-parameters/settings. """

    if model_type == 'encdec':
        from seq2seq.models.encdec import EncoderDecoder
        model = EncoderDecoder(**kwargs)
    else:
        raise ValueError('Unknown model type')

    if use_cuda:
        model = model.cuda()

    return model


def visdom_plot(stats, eval_every=1, visdom_windows=None, title='untitled'):
    """
    Plot all metrics for a given statistics object.
    Only plots a metric if the object has values for it.
    """
    plot_single_point(stats=stats, metric="loss", eval_every=eval_every,
                      visdom_windows=visdom_windows, title=title)
    plot_single_point(stats=stats, metric="perplexity", eval_every=eval_every,
                      visdom_windows=visdom_windows, title=title)
    plot_single_point(stats=stats, metric="accuracy", eval_every=eval_every,
                      visdom_windows=visdom_windows, title=title)
    plot_single_point(stats=stats, metric="exact_match", eval_every=eval_every,
                      visdom_windows=visdom_windows, title=title)
    plot_single_point(stats=stats, metric="bleu", eval_every=eval_every,
                      visdom_windows=visdom_windows, title=title)


def plot_single_point(stats=None, metric=None, eval_every=1, visdom_windows=None, title='untitled'):
    """ Add a point to a new or existing plot, depending on if there is a Visdom window already. """
    values = getattr(stats, metric)

    if len(values) == 0:
        return

    x = np.array([len(values) * eval_every])
    y = np.array([values[-1]])
    window_name = '%s_%s' % (stats.name, metric)
    win = visdom_windows[window_name]
    update = None if win is None else 'append'
    plot_title = title + " " + stats.name + " " + metric
    win = viz.line(X=x, Y=y, win=win, update=update, opts=dict(
        showlegend=False, markers=True, title=plot_title, xlabel="Iteration", ylabel=metric))

    # remember this window for the next point
    if visdom_windows[window_name] is None:
        visdom_windows[window_name] = win


def plot_single_point_simple(x, y, metric='', window='', title='untitled', visdom_windows=None):

    x = np.array([x])
    y = np.array([y])
    win = visdom_windows[title]
    update = None if win is None else 'append'
    win = viz.line(X=x, Y=y, win=win, update=update, opts=dict(
        showlegend=False, markers=True, title=title, xlabel="Iteration", ylabel=metric))

    # remember this window for the next point
    if visdom_windows[title] is None:
        visdom_windows[title] = win


def plot_heatmap_simple(scores, path='plot.png', title="", fontsize=10, columnnames=None, rownames=None):

    font = {'size': fontsize}
    matplotlib.rc('font', **font)

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(scores, cmap='viridis', vmin=0., vmax=1.)

    scores = scores[::-1]

    # set tick labels to the words
    ax.set_xticklabels(columnnames, minor=False, rotation=90)
    ax.set_yticklabels(rownames, minor=False, rotation=0)

    ax.set_aspect(1)
    ax.set_title(title, y=1.02)

    # put the major ticks at the middle of each cell
    # and the x-ticks on top
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(scores.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(scores.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()

    # plt.colorbar(heatmap)
    if path.endswith('.pdf'):
        pp = PdfPages(path)
        pp.savefig()
        pp.close()
    else:
        plt.savefig(path, dpi=300)
        plt.close('all')


def animate_images(output_path=None, filenames=[], duration=0.5):
    """
    Animate images, e.g. heatmap over time

    Args:
        output_path: path to .gif output
        filenames:

    Returns:

    """
    assert output_path.endswith('.gif'), "output must be .gif"
    with imageio.get_writer(output_path, mode='I', duration=duration) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
