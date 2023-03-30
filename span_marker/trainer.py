from typing import Callable, Dict, List, Optional, Tuple

import evaluate
import torch
from datasets import Dataset
from transformers import (
    EvalPrediction,
    PreTrainedModel,
    TrainerCallback,
    TrainingArguments,
)
from transformers import (
    Trainer as TransformersTrainer,
)

from span_marker.data.label_normalizer import AutoLabelNormalizer
from span_marker.modeling import SpanMarkerModel
from span_marker.tokenizer import SpanMarkerTokenizer


def compute_f1_via_seqeval(tokenizer: SpanMarkerTokenizer, eval_prediction: EvalPrediction) -> Dict[str, float]:
    """Compute micro-F1, recall, precision and accuracy scores using `seqeval` for the evaluation predictions.

    Note:
        We assume that samples are not shuffled for the evaluation/prediction.
        With other words, don't use this on the (shuffled) train dataset!

    Args:
        tokenizer (SpanMarkerTokenizer):
        eval_prediction (EvalPrediction): _description_

    Returns:
        Dict[str, float]: Dictionary with `"overall_precision"`, `"overall_recall"`, `"overall_f1"`
            and `"overall_accuracy"` keys.
    """
    inputs = eval_prediction.inputs
    gold_labels = eval_prediction.label_ids
    logits = eval_prediction.predictions[0]
    num_words = eval_prediction.predictions[-1]

    # Compute probabilities via softmax and extract 'winning' scores/labels
    probs = torch.tensor(logits).softmax(dim=-1)
    scores, pred_labels = probs.max(-1)

    # Collect all samples in one dict. We do this because some samples are spread between multiple inputs
    sample_dict = {}
    for sample_idx in range(inputs.shape[0]):
        tokens = inputs[sample_idx]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        token_hash = hash(text)
        if token_hash not in sample_dict:
            spans = list(tokenizer.get_all_valid_spans(num_words[sample_idx], tokenizer.config.entity_max_length))
            sample_dict[token_hash] = {
                "text": text,
                "spans": spans,
                "gold_labels": gold_labels[sample_idx].tolist(),
                "pred_labels": pred_labels[sample_idx].tolist(),
                "scores": scores[sample_idx].tolist(),
                "num_words": num_words[sample_idx],
            }
        else:
            sample_dict[token_hash]["gold_labels"] += gold_labels[sample_idx].tolist()
            sample_dict[token_hash]["pred_labels"] += pred_labels[sample_idx].tolist()
            sample_dict[token_hash]["scores"] += scores[sample_idx].tolist()

    outside_id = tokenizer.config.outside_id
    id2label = {int(label_id): label for label_id, label in tokenizer.config.id2label.items()}
    if tokenizer.config.are_labels_schemed():
        id2label = {label_id: id2label[tokenizer.config.id2reduced_id[label_id]] for label_id in id2label}
    # seqeval works wonders for NER evaluation
    seqeval = evaluate.load("seqeval")
    for sample in sample_dict.values():
        spans = sample["spans"]
        scores = sample["scores"]
        num_words = sample["num_words"]
        gold_labels = sample["gold_labels"]
        pred_labels = sample["pred_labels"]

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

    return seqeval.compute()


class Trainer(TransformersTrainer):
    def __init__(
        self,
        model: SpanMarkerModel = None,
        args: TrainingArguments = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ) -> None:
        # Extract the model from an initializer function
        if model_init:
            self.model_init = model_init
            model = self.call_model_init()

        # Convert dataset labels to a common format (list of label-start-end tuples)
        label_normalizer = AutoLabelNormalizer.from_config(model.config)

        if train_dataset:
            train_dataset = train_dataset.map(label_normalizer, input_columns="ner_tags", batched=True)
            # Tokenize and add start/end markers
            train_dataset = train_dataset.map(
                lambda batch: model.tokenizer(batch["tokens"], labels=batch["ner_tags"]),
                batched=True,
                remove_columns=train_dataset.column_names,
            )
        if eval_dataset:
            eval_dataset = eval_dataset.map(label_normalizer, input_columns="ner_tags", batched=True)
            # Tokenize and add start/end markers, return tokens for use in the metrics computations
            eval_dataset = eval_dataset.map(
                lambda batch: model.tokenizer(batch["tokens"], labels=batch["ner_tags"], is_evaluate=True),
                batched=True,
                remove_columns=eval_dataset.column_names,
            )

        # Set some Training arguments that must be set for SpanMarker
        if args is None:
            args = TrainingArguments()
        args.include_inputs_for_metrics = True
        args.remove_unused_columns = False

        # Always compute `compute_f1_via_seqeval` - optionally compute user-provided metrics
        if compute_metrics is not None:
            compute_metrics = lambda eval_prediction: {
                **compute_f1_via_seqeval(model.tokenizer, eval_prediction),
                **compute_metrics(eval_prediction),
            }
        else:
            compute_metrics = lambda eval_prediction: compute_f1_via_seqeval(model.tokenizer, eval_prediction)

        super().__init__(
            model=model,
            args=args,
            data_collator=model.data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=model.tokenizer,
            model_init=None,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        # We have to provide the __init__ with None for model_init and then override it here again
        # We do this because we need `model` to already be defined in this SpanMarker Trainer class
        # and the Transformers Trainer would complain if we provide both a model and a model_init
        # in its __init__.
        self.model_init = model_init
