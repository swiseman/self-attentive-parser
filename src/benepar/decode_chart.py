"""
Parsing formulated as span classification (https://arxiv.org/abs/1705.03919)
"""

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_struct

from .parse_base import CompressedParserOutput


def pad_charts(charts, padding_value=-100):
    """Pad a list of variable-length charts with `padding_value`."""
    batch_size = len(charts)
    max_len = max(chart.shape[0] for chart in charts)
    padded_charts = torch.full(
        (batch_size, max_len, max_len),
        padding_value,
        dtype=charts[0].dtype,
        device=charts[0].device,
    )
    for i, chart in enumerate(charts):
        chart_size = chart.shape[0]
        padded_charts[i, :chart_size, :chart_size] = chart
    return padded_charts

def pad_charts2(charts, nlabels, padding_value=-100):
    """Pad a list of variable-length charts with `padding_value`."""
    batch_size = len(charts)
    max_len = max(chart.shape[0] for chart in charts)
    padded_charts = torch.full(
        (batch_size, max_len, max_len, nlabels),
        padding_value,
        dtype=charts[0].dtype,
        device=charts[0].device)
    for i, chart in enumerate(charts):
        chart_size = chart.shape[0]
        padded_charts[i, :chart_size, :chart_size] = chart
    return padded_charts

def collapse_unary_strip_pos(tree, strip_top=True):
    """Collapse unary chains and strip part of speech tags."""

    def strip_pos(tree):
        if len(tree) == 1 and isinstance(tree[0], str):
            return tree[0]
        else:
            return nltk.tree.Tree(tree.label(), [strip_pos(child) for child in tree])

    collapsed_tree = strip_pos(tree)
    collapsed_tree.collapse_unary(collapsePOS=True, joinChar="::")
    if collapsed_tree.label() in ("TOP", "ROOT", "S1", "VROOT"):
        if strip_top:
            if len(collapsed_tree) == 1:
                collapsed_tree = collapsed_tree[0]
            else:
                collapsed_tree.set_label("")
        elif len(collapsed_tree) == 1:
            collapsed_tree[0].set_label(
                collapsed_tree.label() + "::" + collapsed_tree[0].label())
            collapsed_tree = collapsed_tree[0]
    return collapsed_tree


def _get_labeled_spans(tree, spans_out, start):
    if isinstance(tree, str):
        return start + 1

    assert len(tree) > 1 or isinstance(
        tree[0], str
    ), "Must call collapse_unary_strip_pos first"
    end = start
    for child in tree:
        end = _get_labeled_spans(child, spans_out, end)
    # Spans are returned as closed intervals on both ends
    spans_out.append((start, end - 1, tree.label()))
    return end


def get_labeled_spans(tree):
    """Converts a tree into a list of labeled spans.

    Args:
        tree: an nltk.tree.Tree object

    Returns:
        A list of (span_start, span_end, span_label) tuples. The start and end
        indices indicate the first and last words of the span (a closed
        interval). Unary chains are collapsed, so e.g. a (S (VP ...)) will
        result in a single span labeled "S+VP".
    """
    tree = collapse_unary_strip_pos(tree)
    spans_out = []
    _get_labeled_spans(tree, spans_out, start=0)
    return spans_out


def uncollapse_unary(tree, ensure_top=False):
    """Un-collapse unary chains."""
    if isinstance(tree, str):
        return tree
    else:
        labels = tree.label().split("::")
        if ensure_top and labels[0] != "TOP":
            labels = ["TOP"] + labels
        children = []
        for child in tree:
            child = uncollapse_unary(child)
            children.append(child)
        for label in labels[::-1]:
            children = [nltk.tree.Tree(label, children)]
        return children[0]

# based on nltk's _pformat_flat
def my_pformat_flat(tree, parens="()", closing_label=True, dummy_word="X"):
    nodesep, quotes = "", False
    childstrs = []
    for child in tree:
        if isinstance(child, nltk.tree.Tree):
            childstrs.append(my_pformat_flat(child, parens, closing_label, dummy_word))
        elif isinstance(child, tuple):
            childstrs.append("/".join(child))
        elif isinstance(child, str) and not quotes:
            childwrd = dummy_word if dummy_word is not None else child
            childstrs.append("%s" % childwrd)
        else:
            childstrs.append(repr(child))
    if isinstance(tree._label, str):
        closelabel = tree._label if closing_label else ""
        return "{}{}{} {} {}{}".format(
            parens[0], tree._label, nodesep, " ".join(childstrs), closelabel, parens[1])
    closelabel = repr(tree._label) if closing_label else ""
    return "{}{}{} {} {}{}".format(
        parens[0], repr(tree._label), nodesep, " ".join(childstrs), closelabel, parens[1])


class ChartDecoder:
    """A chart decoder for parsing formulated as span classification."""

    def __init__(self, label_vocab, force_root_constituent=True):
        """Constructs a new ChartDecoder object.
        Args:
            label_vocab: A mapping from span labels to integer indices.
        """
        self.label_vocab = label_vocab
        self.label_from_index = {i: label for label, i in label_vocab.items()}
        self.force_root_constituent = force_root_constituent

    @staticmethod
    def build_vocab(trees):
        label_set = set()
        for tree in trees:
            for _, _, label in get_labeled_spans(tree):
                if label:
                    label_set.add(label)
        label_set = [""] + sorted(label_set)
        return {label: i for i, label in enumerate(label_set)}

    @staticmethod
    def infer_force_root_constituent(trees):
        for tree in trees:
            for _, _, label in get_labeled_spans(tree):
                if not label:
                    return False
        return True

    def chart_from_tree(self, tree):
        spans = get_labeled_spans(tree)
        num_words = len(tree.leaves())
        chart = np.full((num_words, num_words), -100, dtype=int)
        chart = np.tril(chart, -1)
        # Now all invalid entries are filled with -100, and valid entries with 0
        for start, end, label in spans:
            # Previously unseen unary chains can occur in the dev/test sets.
            # For now, we ignore them and don't mark the corresponding chart
            # entry as a constituent.
            if label in self.label_vocab:
                chart[start, end] = self.label_vocab[label]
        return chart

    def chart_from_tree2(self, tree):
        spans = get_labeled_spans(tree)
        num_words = len(tree.leaves())
        chart = torch.zeros(num_words, num_words, len(self.label_vocab))
        chart[..., 0] = 1 # by default predict no span
        # -100s on the lower triangle, zeros elsewhere
        trilidxs = torch.tril_indices(num_words, num_words, offset=-1, device=chart.device)
        chart[trilidxs[0], trilidxs[1]] = -100
        # chart = torch.full((num_words, num_words), -100).tril(-1).unsqueeze(2).expand(
        #     num_words, num_words, len(self.label_vocab)).contiguous()
        for start, end, label in spans:
            # Previously unseen unary chains can occur in the dev/test sets.
            # For now, we ignore them and don't mark the corresponding chart
            # entry as a constituent.
            if label in self.label_vocab:
                chart[start, end, self.label_vocab[label]] = 1.0
                chart[start, end, 0] = 0.0
        return chart

    def chart_from_tree3(self, tree):
        spans = get_labeled_spans(tree)
        num_words = len(tree.leaves())
        chart = np.zeros((num_words, num_words), dtype=int)
        trilidxs = np.tril_indices(num_words, k=-1)
        chart[trilidxs] = -100
        # Now all invalid entries are filled with -100, and valid entries with 0
        for start, end, label in spans:
            # Previously unseen unary chains can occur in the dev/test sets.
            # For now, we ignore them and don't mark the corresponding chart
            # entry as a constituent.
            if label in self.label_vocab:
                chart[start, end] = self.label_vocab[label]
        return chart

    def charts_from_pytorch_scores_batched(self, scores, lengths):
        """Runs CKY to recover span labels from scores (e.g. logits).

        This method uses pytorch-struct to speed up decoding compared to the
        pure-Python implementation of CKY used by tree_from_scores().

        Args:
            scores: a pytorch tensor of shape (batch size, max length,
                max length, label vocab size).
            lengths: a pytorch tensor of shape (batch size,)

        Returns:
            A list of numpy arrays, each of shape (sentence length, sentence
                length).
        """
        scores = scores.detach()
        scores = scores - scores[..., :1]
        if self.force_root_constituent:
            scores[torch.arange(scores.shape[0]), 0, lengths - 1, 0] -= 1e9
        dist = torch_struct.TreeCRF(scores, lengths=lengths)
        amax = dist.argmax
        amax[..., 0] += 1e-9
        padded_charts = amax.argmax(-1)
        padded_charts = padded_charts.detach().cpu().numpy()
        return [
            chart[:length, :length] for chart, length in zip(padded_charts, lengths)
        ]

    def charts_from_pytorch_scores_batched2(self, scores, lengths, thresh=0.0):
        # import ipdb; ipdb.set_trace()
        scores = scores.detach()
        #scores = scores - scores[..., :1]
        if self.force_root_constituent:
            scores[torch.arange(scores.shape[0]), 0, lengths - 1, 0] = thresh - 1.0

        # get rid of padding and tril stuff
        maxlen = scores.size(1)
        arng = torch.arange(maxlen, device=lengths.device).view(1, -1)
        is_pad = arng >= lengths.view(-1, 1)
        scores[is_pad.unsqueeze(2) | is_pad.unsqueeze(1)] = -float("inf")
        scores[(arng.view(-1, 1) > arng).unsqueeze(0).unsqueeze(-1).expand(
            scores.size())] = -float("inf")

        # for now we'll take highest thing assuming it's higher than thresh?
        maxes, argmaxes = scores.max(-1)
        argmaxes[maxes <= thresh] = 0
        padded_charts = argmaxes.cpu().numpy()
        return [
            chart[:length, :length] for chart, length in zip(padded_charts, lengths)
        ]

    def charts_from_pytorch_scores_batched3(self, scores, lengths):
        scores = scores.detach()
        #scores = scores - scores[..., :1]
        if self.force_root_constituent:
            scores[torch.arange(scores.shape[0]), 0, lengths - 1, 0] -= 1.0 # made up

        maxes, argmaxes = scores.max(-1)
        # get rid of padding and tril stuff
        maxlen = scores.size(1)
        arng = torch.arange(maxlen, device=lengths.device).view(1, -1)
        is_pad = arng >= lengths.view(-1, 1)
        argmaxes[is_pad.unsqueeze(2) | is_pad.unsqueeze(1)] = 0
        argmaxes[(arng.view(-1, 1) > arng).unsqueeze(0).expand(argmaxes.size())] = 0

        padded_charts = argmaxes.cpu().numpy()
        return [
            chart[:length, :length] for chart, length in zip(padded_charts, lengths)
        ]

    def pants(self, scores, lengths, stop_thresh=0):
        scores = scores.detach()
        #scores = scores - scores[..., :1]
        if self.force_root_constituent:
            scores[torch.arange(scores.shape[0]), 0, lengths - 1, 0] -= 1.0 # made up

        maxes, argmaxes = scores.max(-1) # bsz x T x T
        # get rid of padding and tril stuff
        bsz, maxlen, _ = maxes.size()
        arng = torch.arange(maxlen, device=lengths.device).view(1, -1)
        is_pad = arng >= lengths.view(-1, 1)
        maxes[is_pad.unsqueeze(2) | is_pad.unsqueeze(1)] = -float("inf")
        maxes[(arng.view(-1, 1) > arng).unsqueeze(0).expand(
            maxes.size())] = -float("inf")
        srtd, srtidxs = maxes.view(bsz, -1).sort(axis=-1, descending=True)
        #nleft = lengths.clone()
        outputs = []
        for b in range(bsz):
            if lengths[b].item() == 1:
                spans = np.array([[0, 1, 0]])
                outputs.append(CompressedParserOutput(
                    starts=spans[:, 0], ends=spans[:, 1], labels=spans[:, 2]))
                continue
            spans, parsed = {}, False
            for i in range(srtidxs.size(1)):
                flat_idx = srtidxs[b, i].item()
                l = flat_idx // maxlen
                r = flat_idx % maxlen + 1 # exclusive
                labe = argmaxes[b, l, r-1].item()
                if (labe == 0 or l >= lengths[b].item() or r > lengths[b].item()
                    or (any((lp < l < rp < r) or (l < lp < r < rp)
                       for (lp, rp) in spans))):
                    continue
                if srtd[b, i].item() > -float("inf"): # can sometimes continue until padding
                    spans[l, r] = labe
                parsed = parsed or (l == 0 and r == lengths[b].item())
                if parsed and i < srtidxs.size(1)-1 and srtd[b, i+1].item() < stop_thresh:
                    break
            #if not parsed:
            #    import ipdb; ipdb.set_trace()
            # add diagonal
            spans.update({(j, j+1): 0 for j in range(lengths[b].item())
                          if (j, j+1) not in spans})
            #spans.sort(key=lambda x: (-x[1], x[0])) # maybe better to sort in numpy?
            spans = sorted([k + (v,) for k, v in spans.items()],
                           key=lambda x: (x[0], -x[1])) # NB opposite order from np
            spans = np.array(spans) # first col is starts, then ends, then labels
            outputs.append(CompressedParserOutput(
                starts=spans[:, 0], ends=spans[:, 1], labels=spans[:, 2]))
        assert len(outputs) == bsz
        return outputs

    def compressed_output_from_chart(self, chart):
        chart_with_filled_diagonal = chart.copy()
        np.fill_diagonal(chart_with_filled_diagonal, 1)
        chart_with_filled_diagonal[0, -1] = 1
        starts, inclusive_ends = np.where(chart_with_filled_diagonal)
        preorder_sort = np.lexsort((-inclusive_ends, starts))
        starts = starts[preorder_sort]
        inclusive_ends = inclusive_ends[preorder_sort]
        labels = chart[starts, inclusive_ends]
        ends = inclusive_ends + 1
        return CompressedParserOutput(starts=starts, ends=ends, labels=labels)

    def tree_from_chart(self, chart, leaves):
        compressed_output = self.compressed_output_from_chart(chart)
        return compressed_output.to_tree(leaves, self.label_from_index)

    def tree_from_scores(self, scores, leaves):
        """Runs CKY to decode a tree from scores (e.g. logits).

        If speed is important, consider using charts_from_pytorch_scores_batched
        followed by compressed_output_from_chart or tree_from_chart instead.

        Args:
            scores: a chart of scores (or logits) of shape
                (sentence length, sentence length, label vocab size). The first
                two dimensions may be padded to a longer length, but all padded
                values will be ignored.
            leaves: the leaf nodes to use in the constructed tree. These
                may be of type str or nltk.Tree, or (word, tag) tuples that
                will be used to construct the leaf node objects.

        Returns:
            An nltk.Tree object.
        """
        leaves = [
            nltk.Tree(node[1], [node[0]]) if isinstance(node, tuple) else node
            for node in leaves
        ]

        chart = {}
        scores = scores - scores[:, :, 0, None]
        for length in range(1, len(leaves) + 1):
            for left in range(0, len(leaves) + 1 - length):
                right = left + length

                label_scores = scores[left, right - 1]
                label_scores = label_scores - label_scores[0]

                argmax_label_index = int(
                    label_scores.argmax()
                    if length < len(leaves) or not self.force_root_constituent
                    else label_scores[1:].argmax() + 1
                )
                argmax_label = self.label_from_index[argmax_label_index]
                label = argmax_label
                label_score = label_scores[argmax_label_index]

                if length == 1:
                    tree = leaves[left]
                    if label:
                        tree = nltk.tree.Tree(label, [tree])
                    chart[left, right] = [tree], label_score
                    continue

                best_split = max(
                    range(left + 1, right),
                    key=lambda split: (chart[left, split][1] + chart[split, right][1]),
                )

                left_trees, left_score = chart[left, best_split]
                right_trees, right_score = chart[best_split, right]

                children = left_trees + right_trees
                if label:
                    children = [nltk.tree.Tree(label, children)]

                chart[left, right] = (children, label_score + left_score + right_score)

        children, score = chart[0, len(leaves)]
        tree = nltk.tree.Tree("TOP", children)
        tree = uncollapse_unary(tree)
        return tree


class SpanClassificationMarginLoss(nn.Module):
    def __init__(self, force_root_constituent=True, reduction="mean"):
        super().__init__()
        self.force_root_constituent = force_root_constituent
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"Invalid value for reduction: {reduction}")
        self.reduction = reduction

    def forward(self, logits, labels):
        gold_event = F.one_hot(F.relu(labels), num_classes=logits.shape[-1])

        logits = logits - logits[..., :1]
        lengths = (labels[:, 0, :] != -100).sum(-1)
        augment = (1 - gold_event).to(torch.float)
        if self.force_root_constituent:
            augment[torch.arange(augment.shape[0]), 0, lengths - 1, 0] -= 1e9
        dist = torch_struct.TreeCRF(logits + augment, lengths=lengths)

        pred_score = dist.max
        gold_score = (logits * gold_event).sum((1, 2, 3))

        margin_losses = F.relu(pred_score - gold_score)

        if self.reduction == "none":
            return margin_losses
        elif self.reduction == "mean":
            return margin_losses.mean()
        elif self.reduction == "sum":
            return margin_losses.sum()
        else:
            assert False, f"Unexpected reduction: {self.reduction}"
