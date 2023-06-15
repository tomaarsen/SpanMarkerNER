import warnings
from typing import Dict

import evaluate
import torch
from sklearn.exceptions import UndefinedMetricWarning
from transformers import EvalPrediction

from span_marker.tokenizer import SpanMarkerTokenizer


def compute_f1_via_seqeval(tokenizer: SpanMarkerTokenizer, eval_prediction: EvalPrediction) -> Dict[str, float]:
    """Compute micro-F1, recall, precision and accuracy scores using ``seqeval`` for the evaluation predictions.

    Note:
        We assume that samples are not shuffled for the evaluation/prediction.
        With other words, don't use this on the (shuffled) train dataset!

    Args:
        tokenizer (SpanMarkerTokenizer): The model its tokenizer.
        eval_prediction (~transformers.EvalPrediction): The predictions resulting from the evaluations.

    Returns:
        Dict[str, float]: Dictionary with ``"overall_precision"``, ``"overall_recall"``, ``"overall_f1"``
            and ``"overall_accuracy"`` keys.
    """
    inputs = eval_prediction.inputs
    gold_labels = eval_prediction.label_ids
    logits = eval_prediction.predictions[0]
    num_words = eval_prediction.predictions[2]
    has_document_context = len(eval_prediction.predictions) == 5
    if has_document_context:
        document_ids = eval_prediction.predictions[3]
        sentence_ids = eval_prediction.predictions[4]

    # Compute probabilities via softmax and extract 'winning' scores/labels
    probs = torch.tensor(logits, dtype=torch.float32).softmax(dim=-1)
    scores, pred_labels = probs.max(-1)

    # Collect all samples in one dict. We do this because some samples are spread between multiple inputs
    sample_list = []
    for sample_idx in range(inputs.shape[0]):
        tokens = inputs[sample_idx]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        token_hash = hash(text) if not has_document_context else (document_ids[sample_idx], sentence_ids[sample_idx])
        if (
            not sample_list
            or sample_list[-1]["hash"] != token_hash
            or len(sample_list[-1]["spans"]) == len(sample_list[-1]["gold_labels"])
        ):
            mask = gold_labels[sample_idx] != -100
            spans = list(tokenizer.get_all_valid_spans(num_words[sample_idx], tokenizer.config.entity_max_length))
            sample_list.append(
                {
                    "text": text,
                    "gold_labels": gold_labels[sample_idx][mask].tolist(),
                    "pred_labels": pred_labels[sample_idx][mask].tolist(),
                    "scores": scores[sample_idx].tolist(),
                    "num_words": num_words[sample_idx],
                    "hash": token_hash,
                    "spans": spans,
                }
            )
        else:
            mask = gold_labels[sample_idx] != -100
            sample_list[-1]["gold_labels"] += gold_labels[sample_idx][mask].tolist()
            sample_list[-1]["pred_labels"] += pred_labels[sample_idx][mask].tolist()
            sample_list[-1]["scores"] += scores[sample_idx].tolist()

    outside_id = tokenizer.config.outside_id
    id2label = tokenizer.config.id2label
    # seqeval works wonders for NER evaluation
    seqeval = evaluate.load("seqeval")
    for sample in sample_list:
        scores = sample["scores"]
        num_words = sample["num_words"]
        spans = sample["spans"]
        gold_labels = sample["gold_labels"]
        pred_labels = sample["pred_labels"]
        assert len(gold_labels) == len(pred_labels) and len(spans) == len(pred_labels)

        # Construct IOB2 format for gold labels, useful for seqeval
        gold_labels_per_tokens = ["O"] * num_words
        for span, gold_label in zip(spans, gold_labels):
            if gold_label != outside_id:
                gold_labels_per_tokens[span[0]] = "B-" + id2label[gold_label]
                gold_labels_per_tokens[span[0] + 1 : span[1]] = ["I-" + id2label[gold_label]] * (span[1] - span[0] - 1)

        # Same for predictions, note that we place most likely spans first and we disallow overlapping spans for now.
        pred_labels_per_tokens = ["O"] * num_words
        for _, span, pred_label in sorted(zip(scores, spans, pred_labels), key=lambda tup: tup[0], reverse=True):
            if pred_label != outside_id and all(pred_labels_per_tokens[i] == "O" for i in range(span[0], span[1])):
                pred_labels_per_tokens[span[0]] = "B-" + id2label[pred_label]
                pred_labels_per_tokens[span[0] + 1 : span[1]] = ["I-" + id2label[pred_label]] * (span[1] - span[0] - 1)

        seqeval.add(prediction=pred_labels_per_tokens, reference=gold_labels_per_tokens)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        results = seqeval.compute()
    # `results` also contains e.g. "person-athlete": {'precision': 0.5982658959537572, 'recall': 0.9, 'f1': 0.71875, 'number': 230}
    # logging this all is overkill. Tensorboard doesn't even support it, WandB does, but it's not very useful generally.
    # I'd like to revisit this to expose this information somehow still
    return {key: value for key, value in results.items() if isinstance(value, float)}
