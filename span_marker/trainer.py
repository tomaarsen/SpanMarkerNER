from typing import Callable, Dict, List, Optional, Tuple

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

from span_marker.label_normalizer import AutoLabelNormalizer, LabelNormalizer
from span_marker.modeling import SpanMarkerModel
from span_marker.tokenizer import SpanMarkerTokenizer


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

        # To convert dataset labels to a common format (list of label-start-end tuples)
        label_normalizer = AutoLabelNormalizer.from_config(model.config)
        # Normalize labels & tokenize the provided datasets
        if train_dataset:
            train_dataset = self.preprocess_dataset(train_dataset, label_normalizer, model.tokenizer)
        if eval_dataset:
            eval_dataset = self.preprocess_dataset(
                eval_dataset, label_normalizer, model.tokenizer, dataset_name="eval", is_evaluate=True
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

    def preprocess_dataset(
        self,
        dataset: Dataset,
        label_normalizer: LabelNormalizer,
        tokenizer: SpanMarkerTokenizer,
        dataset_name: str = "train",
        is_evaluate: bool = False,
    ) -> Dataset:
        for column in ("tokens", "ner_tags"):
            if column not in dataset.column_names:
                raise ValueError(f"The {dataset_name} dataset must contain a {column!r} column.")

        # Normalize the labels to a common format (list of label-start-end tuples)
        dataset = dataset.map(
            label_normalizer,
            input_columns="ner_tags",
            batched=True,
            desc=f"Label normalizing the {dataset_name} dataset",
        )
        # Tokenize and add start/end markers
        dataset = dataset.map(
            lambda batch: tokenizer(batch["tokens"], labels=batch["ner_tags"], is_evaluate=is_evaluate),
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing the {dataset_name} dataset",
        )
        return dataset
