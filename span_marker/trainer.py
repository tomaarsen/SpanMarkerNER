import warnings
from typing import Callable, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    EvalPrediction,
    TrainerCallback,
    TrainingArguments,
)
from transformers import Trainer as TransformersTrainer
from transformers.trainer_utils import PredictionOutput

from span_marker.evaluation import compute_f1_via_seqeval
from span_marker.label_normalizer import AutoLabelNormalizer, LabelNormalizer
from span_marker.modeling import SpanMarkerModel
from span_marker.tokenizer import SpanMarkerTokenizer


class Trainer(TransformersTrainer):
    """
    Trainer is a simple but feature-complete training and eval loop for SpanMarker,
    built tightly on top of the 🤗 Transformers :external:doc:`Trainer <main_classes/trainer>`.

    Args:
        model (Optional[SpanMarkerModel]):
            The model to train, evaluate or use for predictions. If not provided, a ``model_init`` must be passed.
        args (Optional[~transformers.TrainingArguments]):
            The arguments to tweak for training. Will default to a basic instance of :class:`~transformers.TrainingArguments` with the
            ``output_dir`` set to a directory named *models/my_span_marker_model* in the current directory if not provided.
        train_dataset (Optional[~datasets.Dataset]):
            The dataset to use for training.
        eval_dataset (Optional[~datasets.Dataset]):
             The dataset to use for evaluation.
        model_init (Optional[Callable[[], SpanMarkerModel]]):
            A function that instantiates the model to be used. If provided, each call to :meth:`Trainer.train` will start
            from a new instance of the model as given by this function.

            The function may have zero argument, or a single one containing the optuna/Ray Tune/SigOpt trial object, to
            be able to choose different architectures according to hyper parameters (such as layer count, sizes of
            inner layers, dropout probabilities etc).
        compute_metrics (Optional[Callable[[~transformers.EvalPrediction], Dict]]):
            The function that will be used to compute metrics at evaluation. Must take a :class:`~transformers.EvalPrediction` and return
            a dictionary string to metric values.
        callbacks (Optional[List[~transformers.TrainerCallback]]):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in the Hugging Face :external:doc:`Callback documentation <main_classes/callback>`.

            If you want to remove one of the default callbacks used, use the :meth:`~Trainer.remove_callback` method.
        optimizers (Tuple[Optional[~torch.optim.Optimizer], Optional[~torch.optim.lr_scheduler.LambdaLR]]): A tuple
            containing the optimizer and the scheduler to use. Will default to an instance of ``AdamW`` on your model
            and a scheduler given by ``get_linear_schedule_with_warmup`` controlled by ``args``.
        preprocess_logits_for_metrics (Optional[Callable[[~torch.Tensor, ~torch.Tensor], ~torch.Tensor]]):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by ``compute_metrics``.

            Note that the labels (second parameter) will be ``None`` if the dataset does not have them.

    Important attributes:

        - **model** -- Always points to the core model.
        - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
          original model. This is the model that should be used for the forward pass. For example, under ``DeepSpeed``,
          the inner model is wrapped in ``DeepSpeed`` and then again in :class:`torch.nn.DistributedDataParallel`. If the
          inner model hasn't been wrapped, then ``self.model_wrapped`` is the same as ``self.model``.
        - **is_model_parallel** -- Whether or not a model has been switched to a model parallel mode (different from
          data parallelism, this means some of the model layers are split on different GPUs).
        - **place_model_on_device** -- Whether or not to automatically place the model on the device - it will be set
          to ``False`` if model parallel or deepspeed is used, or if the default
          `TrainingArguments.place_model_on_device` is overridden to return ``False``.
        - **is_in_train** -- Whether or not a model is currently running :meth:`~Trainer.train` (e.g. when ``evaluate`` is called while
          in ``train``)
    """

    def __init__(
        self,
        model: Optional[SpanMarkerModel] = None,
        args: Optional[TrainingArguments] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        model_init: Callable[[], SpanMarkerModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        # Extract the model from an initializer function
        if model_init:
            self.model_init = model_init
            model = self.call_model_init()

        # To convert dataset labels to a common format (list of label-start-end tuples)
        self.label_normalizer = AutoLabelNormalizer.from_config(model.config)

        # Set some Training arguments that must be set for SpanMarker
        if args is None:
            args = TrainingArguments(output_dir="models/my_span_marker_model")
        args.include_inputs_for_metrics = True
        args.remove_unused_columns = False

        # Always compute `compute_f1_via_seqeval` - optionally compute user-provided metrics
        if compute_metrics is not None:
            compute_metrics_func = lambda eval_prediction: {
                **compute_f1_via_seqeval(model.tokenizer, eval_prediction),
                **compute_metrics(eval_prediction),
            }
        else:
            compute_metrics_func = lambda eval_prediction: compute_f1_via_seqeval(model.tokenizer, eval_prediction)

        super().__init__(
            model=model,
            args=args,
            data_collator=model.data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=model.tokenizer,
            model_init=None,
            compute_metrics=compute_metrics_func,
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
        """Normalize the ``ner_tags`` labels and call tokenizer on ``tokens``.

        Args:
            dataset (~datasets.Dataset): A Hugging Face dataset with ``tokens`` and ``ner_tags`` columns.
            label_normalizer (LabelNormalizer): A callable that normalizes ``ner_tags`` into start-end-label tuples.
            tokenizer (SpanMarkerTokenizer): The tokenizer responsible for tokenizing ``tokens`` into input IDs,
                and adding start and end markers.
            dataset_name (str, optional): The name of the dataset. Defaults to "train".
            is_evaluate (bool, optional): Whether to return the number of words for each sample.
                Required for evaluation. Defaults to False.

        Raises:
            ValueError: If the ``dataset`` does not contain ``tokens`` and ``ner_tags`` columns.

        Returns:
            Dataset: The normalized and tokenized version of the input dataset.
        """
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
            lambda batch: tokenizer(batch["tokens"], labels=batch["ner_tags"], return_num_words=is_evaluate),
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing the {dataset_name} dataset",
        )
        return dataset

    def get_train_dataloader(self) -> DataLoader:
        """Return the preprocessed training DataLoader."""
        self.train_dataset = self.preprocess_dataset(self.train_dataset, self.label_normalizer, self.tokenizer)
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """Return the preprocessed evaluation DataLoader."""
        eval_dataset = eval_dataset or self.eval_dataset
        if eval_dataset is not None:
            eval_dataset = self.preprocess_dataset(
                eval_dataset, self.label_normalizer, self.tokenizer, dataset_name="evaluation", is_evaluate=True
            )
        return super().get_eval_dataloader(eval_dataset)

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """Return the preprocessed evaluation DataLoader."""
        test_dataset = self.preprocess_dataset(
            test_dataset, self.label_normalizer, self.tokenizer, dataset_name="test", is_evaluate=True
        )
        return super().get_test_dataloader(test_dataset)

    def predict(
        self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        warnings.warn(
            f"`Trainer.predict` is not recommended for a {self.model.__class__.__name__}. "
            f"Consider using `{self.model.__class__.__name__}.predict` instead.",
            UserWarning,
            stacklevel=2,
        )
        return super().predict(test_dataset, ignore_keys, metric_key_prefix)
