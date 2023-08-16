import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import datasets
import tokenizers
import torch
import transformers
from huggingface_hub import CardData, ModelCard, dataset_info, model_info
from huggingface_hub.repocard_data import EvalResult, eval_results_to_model_index
from huggingface_hub.utils import yaml_dump
from transformers import TrainerCallback
from transformers.modelcard import (
    extract_hyperparameters_from_trainer,
    make_markdown_table,
)
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

import span_marker

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from span_marker.modeling import SpanMarkerModel
    from span_marker.trainer import Trainer


class ModelCardCallback(TrainerCallback):
    """
    TODO:
    - 1. Label set with 3 examples of each label
    - 2. Training set metrics:
      - Minimum, median, maximum number of words in the training set
      - Minimum, median, maximum number of entities in the training set
      - 3 short example sentences with their tags
    - 3. Ensure metadata is correct and complete
    """

    def __init__(self, trainer: "Trainer") -> None:
        super().__init__()
        self.trainer = trainer

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: "SpanMarkerModel", **kwargs
    ):
        model.model_card_data.hyperparameters = extract_hyperparameters_from_trainer(self.trainer)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: "SpanMarkerModel",
        metrics: Dict[str, float],
        **kwargs,
    ):
        # Set the most recent evaluation scores for the metadata
        model.model_card_data.eval_results_dict = metrics

        if self.trainer.is_in_train:
            # Either set mid-training evaluation metrics
            if "eval_loss" in metrics:
                model.model_card_data.eval_lines_list.append(
                    {
                        # "Training Loss": self.state.log_history[-1]["loss"] if "loss" in self.state.log_history[-1] else "-",
                        "Epoch": state.epoch,
                        "Step": state.global_step,
                        "Validation Loss": metrics["eval_loss"],
                        "Validation Precision": metrics["eval_overall_precision"],
                        "Validation Recall": metrics["eval_overall_recall"],
                        "Validation F1": metrics["eval_overall_f1"],
                        "Validation Accuracy": metrics["eval_overall_accuracy"],
                    }
                )
        else:
            # Or set the post-training metrics
            # Determine the dataset split
            runtime_key = [key for key in metrics.keys() if key.endswith("_runtime")]
            if not runtime_key:
                return
            dataset_split = runtime_key[0][: -len("_runtime")]

            metric_lines = []
            for key, value in metrics.items():
                if not isinstance(value, float):
                    metric_lines.append(
                        {
                            "Label": key[len(dataset_split) + 1 :],
                            "Precision": value["precision"],
                            "Recall": value["recall"],
                            "F1": value["f1"],
                        }
                    )
            metric_lines.insert(
                0,
                {
                    "Label": "**all**",
                    "Precision": metrics[f"{dataset_split}_overall_precision"],
                    "Recall": metrics[f"{dataset_split}_overall_recall"],
                    "F1": metrics[f"{dataset_split}_overall_f1"],
                },
            )
            model.model_card_data.metric_lines = metric_lines


YAML_FIELDS = [
    "language",
    "license",
    "library_name",
    "tags",
    "datasets",
    "metrics",
    "pipeline_tag",
    "widget",
    "model-index",
]
IGNORED_FIELDS = ["model"]


@dataclass
class SpanMarkerModelCardData(CardData):
    # Potentially provided by the user
    language: Optional[Union[str, List[str]]] = None
    license: Optional[str] = None
    tags: Optional[List[str]] = field(
        default_factory=lambda: [
            "span-marker",
            "token-classification",
            "ner",
            "named-entity-recognition",
            "generated_from_span_marker_trainer",
        ]
    )
    model_name: str = "SpanMarker"
    model_id: Optional[str] = None
    encoder_name: Optional[str] = None
    encoder_id: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_revision: Optional[str] = None
    task_name: str = "Named Entity Recognition"

    # Automatically filled by `ModelCardCallback` and the Trainer directly
    hyperparameters: Dict[str, Any] = field(default_factory=dict, init=False)
    eval_results_dict: Optional[Dict[str, Any]] = field(default_factory=dict, init=False)
    eval_lines_list: List[Dict[str, float]] = field(default_factory=list, init=False)
    metric_lines: List[Dict[str, float]] = field(default_factory=list, init=False)
    widget: List[Dict[str, str]] = field(default_factory=list, init=False)

    # Computed once, always unchanged
    pipeline_tag: Optional[str] = field(default="token-classification", init=False)
    library_name: Optional[str] = field(default="span-marker", init=False)
    version: Dict[str, str] = field(
        default_factory=lambda: {
            "span_marker": span_marker.__version__,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "datasets": datasets.__version__,
            "tokenizers": tokenizers.__version__,
        },
        init=False,
    )
    metrics: List[str] = field(default_factory=lambda: ["precision", "recall", "f1"], init=False)

    # Passed via `register_model` only
    model: Optional["SpanMarkerModel"] = field(default=None, init=False)

    def __post_init__(self):
        # We don't want to save "ignore_metadata_errors" in our Model Card
        if self.dataset_id and not is_on_huggingface(self.dataset_id, is_model=False):
            logger.warning(
                f"The provided {self.dataset_id!r} dataset could not be found on the Hugging Face Hub."
                " Setting `dataset_id` to None."
            )
            self.dataset_id = None

        if self.encoder_id and not is_on_huggingface(self.encoder_id):
            logger.warning(
                f"The provided {self.encoder_id!r} model could not be found on the Hugging Face Hub."
                " Setting `encoder_id` to None."
            )
            self.encoder_id = None

        if self.model_id and self.model_id.count("/") != 1:
            logger.warning(
                f"The provided {self.model_id!r} model ID should include the organization or user,"
                ' such as "tomaarsen/span-marker-mbert-base-multinerd". Setting `model_id` to None.'
            )
            self.model_id = None

        # TODO: Set model_id based on training args if possible?
        # TODO: Set model_name based on encoder_id/encoder_name and dataset_id/dataset_name?

    def register_model(self, model: "SpanMarkerModel"):
        self.model = model

    def to_dict(self) -> Dict[str, Any]:
        super_dict: dict = super().to_dict()

        # Compute required formats from the raw data
        if self.eval_results_dict:
            dataset_split = list(self.eval_results_dict.keys())[0].split("_")[0]
            eval_results = [
                EvalResult(
                    task_type="token-classification",
                    dataset_type=self.dataset_id,
                    dataset_name=self.dataset_name,
                    metric_type="f1",
                    metric_value=self.eval_results_dict[f"{dataset_split}_overall_f1"],
                    task_name="Named Entity Recognition",
                    dataset_split=dataset_split,
                    dataset_revision=self.dataset_revision,
                    metric_name="F1",
                ),
                EvalResult(
                    task_type="token-classification",
                    dataset_type=self.dataset_id,
                    dataset_name=self.dataset_name,
                    metric_type="precision",
                    metric_value=self.eval_results_dict[f"{dataset_split}_overall_precision"],
                    task_name="Named Entity Recognition",
                    dataset_split=dataset_split,
                    dataset_revision=self.dataset_revision,
                    metric_name="Precision",
                ),
                EvalResult(
                    task_type="token-classification",
                    dataset_type=self.dataset_id,
                    dataset_name=self.dataset_name,
                    metric_type="recall",
                    metric_value=self.eval_results_dict[f"{dataset_split}_overall_recall"],
                    task_name="Named Entity Recognition",
                    dataset_split=dataset_split,
                    dataset_revision=self.dataset_revision,
                    metric_name="Recall",
                ),
            ]
            super_dict["model-index"] = eval_results_to_model_index(self.model_name, eval_results)
        super_dict["eval_lines"] = make_markdown_table(self.eval_lines_list)
        if self.metric_lines:
            super_dict["metrics_table"] = make_markdown_table(self.metric_lines)
        if self.dataset_id:
            super_dict["datasets"] = [self.dataset_id]

        for key in IGNORED_FIELDS:
            super_dict.pop(key, None)
        return {
            **super_dict,
            **self.model.config.to_dict(),
        }

    def to_yaml(self, line_break=None) -> str:
        return yaml_dump(
            {key: value for key, value in self.to_dict().items() if key in YAML_FIELDS},
            sort_keys=False,
            line_break=line_break,
        ).strip()


def is_on_huggingface(repo_id: str, is_model: bool = True) -> bool:
    # Models with more than two 'sections' certainly are not public models
    if len(repo_id.split("/")) > 2:
        return False

    try:
        if is_model:
            model_info(repo_id)
        else:
            dataset_info(repo_id)
        return True
    except:
        # Fetching models can fail for many reasons: Repository not existing, no internet access, HF down, etc.
        return False


def generate_model_card(model: "SpanMarkerModel") -> str:
    template_path = Path(__file__).parent / "model_card_template.md"
    model_card = ModelCard.from_template(card_data=model.model_card_data, template_path=template_path, hf_emoji="🤗")
    return model_card.content

    """
    template = jinja2.Environment().from_string(MODEL_CARD_TEMPLATE)
    save_directory = Path(save_directory)
    context = {}

    context["model_name_or_path"] = "span_marker_model_name"

    if "_name_or_path" in config.encoder:
        context["encoder_name_or_path"] = config.encoder["_name_or_path"]
        context["is_public_model"] = is_public_model(context["encoder_name_or_path"])

    return template.render(context)
    """
