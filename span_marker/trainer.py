import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
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

logger = logging.getLogger(__name__)


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
            The dataset to use for training. Must contain ``tokens`` and ``ner_tags`` columns, and may contain
            ``document_id`` and ``sentence_id`` columns for document-level context during training.
        eval_dataset (Optional[~datasets.Dataset]):
            The dataset to use for evaluation. Must contain ``tokens`` and ``ner_tags`` columns, and may contain
            ``document_id`` and ``sentence_id`` columns for document-level context during evaluation.
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

    REQUIRED_COLUMNS: Tuple[str] = ("tokens", "ner_tags")
    OPTIONAL_COLUMNS: Tuple[str] = ("document_id", "sentence_id")

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

        # Override the type hint
        self.model: SpanMarkerModel

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
        for column in self.REQUIRED_COLUMNS:
            if column not in dataset.column_names:
                raise ValueError(f"The {dataset_name} dataset must contain a {column!r} column.")

        # Drop all unused columns, only keep "tokens", "ner_tags", "document_id", "sentence_id"
        dataset = dataset.remove_columns(
            set(dataset.column_names) - set(self.OPTIONAL_COLUMNS) - set(self.REQUIRED_COLUMNS)
        )
        # Normalize the labels to a common format (list of label-start-end tuples)
        dataset = dataset.map(
            label_normalizer,
            input_columns=("tokens", "ner_tags"),
            desc=f"Label normalizing the {dataset_name} dataset",
        )
        # Tokenize and add start/end markers
        with tokenizer.entity_tracker(split=dataset_name):
            dataset = dataset.map(
                tokenizer,
                batched=True,
                remove_columns=set(dataset.column_names) - set(self.OPTIONAL_COLUMNS),
                desc=f"Tokenizing the {dataset_name} dataset",
                fn_kwargs={"return_num_words": is_evaluate},
            )
        # If "document_id" AND "sentence_id" exist in the training dataset
        if {"document_id", "sentence_id"} <= set(dataset.column_names):
            # If training, set the config flag that this model is trained with document context
            if not is_evaluate:
                self.model.config.trained_with_document_context = True
            # If evaluating and the model was not trained with document context, warn
            elif not self.model.config.trained_with_document_context:
                logger.warning(
                    "This model was trained without document-level context: "
                    "evaluation with document-level context may cause decreased performance."
                )
            dataset = dataset.sort(column_names=["document_id", "sentence_id"])
            dataset = self.add_context(
                dataset,
                tokenizer.model_max_length,
                max_prev_context=self.model.config.max_prev_context,
                max_next_context=self.model.config.max_next_context,
            )
        elif is_evaluate and self.model.config.trained_with_document_context:
            logger.warning(
                "This model was trained with document-level context: "
                "evaluation without document-level context may cause decreased performance."
            )

        # Spread between multiple samples where needed
        original_length = len(dataset)
        dataset = dataset.map(
            Trainer.spread_sample,
            batched=True,
            desc="Spreading data between multiple samples",
            fn_kwargs={
                "model_max_length": tokenizer.model_max_length,
                "marker_max_length": self.model.config.marker_max_length,
            },
        )
        new_length = len(dataset)
        logger.info(
            f"Spread {original_length} sentences across {new_length} samples, "
            f"a {(new_length / original_length) - 1:%} increase. You can increase "
            "`model_max_length` or `marker_max_length` to decrease the number of samples, "
            "but recognize that longer samples are slower."
        )
        return dataset

    @staticmethod
    def add_context(
        dataset: Dataset,
        model_max_length: int,
        max_prev_context: Optional[int] = None,
        max_next_context: Optional[int] = None,
        show_progress_bar: bool = True,
    ) -> Dataset:
        """Add document-level context from previous and next sentences in the same document.

        Args:
            dataset (`Dataset`): The partially processed dataset, containing `"input_ids"`, `"start_position_ids"`,
                `"end_position_ids"`, `"document_id"` and `"sentence_id"` columns.
            model_max_length (`int`): The total number of tokens that can be processed before
                truncation.
            max_prev_context (`Optional[int]`): The maximum number of previous sentences to include. Defaults to None,
                representing as many previous sentences as fits.
            max_next_context (`Optional[int]`): The maximum number of next sentences to include. Defaults to None,
                representing as many previous sentences as fits.
            show_progress_bar (`bool`): Whether to show a progress bar. Defaults to `True`.

        Returns:
            Dataset: A copy of the Dataset with additional previous and next sentences added to input_ids.
        """
        all_input_ids = []
        all_start_position_ids = []
        all_end_position_ids = []
        for sample_idx, sample in tqdm(
            enumerate(dataset),
            desc="Adding document-level context",
            total=len(dataset),
            leave=False,
            disable=not show_progress_bar,
        ):
            # Sequentially add next context, previous context, next context, previous context, etc. until
            # max token length or max_prev/next_context
            tokens = sample["input_ids"][1:-1]
            start_position_ids = sample["start_position_ids"]
            end_position_ids = sample["end_position_ids"]

            next_context_added = 0
            prev_context_added = 0
            remaining_space = model_max_length - len(tokens) - 2
            while remaining_space > 0:
                next_context_index = sample_idx + next_context_added + 1
                should_add_next = (
                    (max_next_context is None or next_context_added < max_next_context)
                    and next_context_index < len(dataset)
                    and dataset[next_context_index]["document_id"] == sample["document_id"]
                )
                if should_add_next:
                    # TODO: [1:-1][:remaining_space] is not efficient
                    tokens += dataset[next_context_index]["input_ids"][1:-1][:remaining_space]
                    next_context_added += 1

                remaining_space = model_max_length - len(tokens) - 2
                if remaining_space <= 0:
                    break

                prev_context_index = sample_idx - prev_context_added - 1
                should_add_prev = (
                    (max_prev_context is None or prev_context_added < max_prev_context)
                    and prev_context_index >= 0
                    and dataset[prev_context_index]["document_id"] == sample["document_id"]
                )
                if should_add_prev:
                    # TODO: [1:-1][remaining_space:] is not efficient
                    prepended_tokens = dataset[prev_context_index]["input_ids"][1:-1][-remaining_space:]
                    tokens = prepended_tokens + tokens
                    # TODO: Use numpy? np.array(sample["start_position_ids"]) + len(prepended_tokens)
                    start_position_ids = [index + len(prepended_tokens) for index in start_position_ids]
                    end_position_ids = [index + len(prepended_tokens) for index in end_position_ids]
                    prev_context_added += 1

                if not should_add_next and not should_add_prev:
                    break

                remaining_space = model_max_length - len(tokens) - 2

            all_input_ids.append([sample["input_ids"][0]] + tokens + [sample["input_ids"][-1]])
            all_start_position_ids.append(start_position_ids)
            all_end_position_ids.append(end_position_ids)

        dataset = dataset.remove_columns(("input_ids", "start_position_ids", "end_position_ids"))
        dataset = dataset.add_column("input_ids", all_input_ids)
        dataset = dataset.add_column("start_position_ids", all_start_position_ids)
        dataset = dataset.add_column("end_position_ids", all_end_position_ids)

        return dataset

    @staticmethod
    def spread_sample(
        batch: Dict[str, List[Any]], model_max_length: int, marker_max_length: int
    ) -> Dict[str, List[Any]]:
        """Spread sentences between multiple samples if lack of space per sample requires it.

        Args:
            batch (`Dict[str, List[Any]]`): A dictionary of dataset keys to lists of values.
            model_max_length (`int`): The total number of tokens that can be processed before
                truncation.
            marker_max_length (`int`): The maximum length for each of the span markers. A value of 128
                means that each training and inferencing sample contains a maximum of 128 start markers
                and 128 end markers, for a total of 256 markers per sample.

        Returns:
            Dict[str, List[Any]]: A dictionary of dataset keys to lists of values.
        """
        keys = batch.keys()
        values = batch.values()
        total_sample_length = model_max_length + 2 * marker_max_length

        batch_samples = {key: [] for key in keys}
        for sample in zip(*values):
            sample = dict(zip(keys, sample))
            sample_marker_space = (total_sample_length - len(sample["input_ids"])) // 2
            spread_between_n = math.ceil(len(sample["start_position_ids"]) / sample_marker_space)
            for i in range(spread_between_n):
                sample_copy = sample.copy()
                start = i * sample_marker_space
                end = (i + 1) * sample_marker_space
                sample_copy["start_position_ids"] = sample["start_position_ids"][start:end]
                sample_copy["end_position_ids"] = sample["end_position_ids"][start:end]
                if "labels" in sample:
                    sample_copy["labels"] = sample["labels"][start:end]
                sample_copy["num_spans"] = len(sample_copy["start_position_ids"])
                for key, value in sample_copy.items():
                    batch_samples[key].append(value)
        return batch_samples

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
        logger.warning(
            f"`Trainer.predict` is not recommended for a {self.model.__class__.__name__}. "
            f"Consider using `{self.model.__class__.__name__}.predict` instead."
        )
        return super().predict(test_dataset, ignore_keys, metric_key_prefix)
