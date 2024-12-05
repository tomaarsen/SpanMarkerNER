import logging
import os
import random
from dataclasses import dataclass, field, fields
from pathlib import Path
from platform import python_version
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import datasets
import tokenizers
import torch
import transformers
from datasets import Dataset
from huggingface_hub import (
    CardData,
    ModelCard,
    dataset_info,
    list_datasets,
    model_info,
)
from huggingface_hub.repocard_data import EvalResult, eval_results_to_model_index
from huggingface_hub.utils import yaml_dump
from transformers import TrainerCallback
from transformers.integrations import CodeCarbonCallback
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
    def __init__(self, trainer: "Trainer") -> None:
        super().__init__()
        self.trainer = trainer

        callbacks = [
            callback for callback in self.trainer.callback_handler.callbacks if isinstance(callback, CodeCarbonCallback)
        ]
        if callbacks:
            trainer.model.model_card_data.code_carbon_callback = callbacks[0]

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
    "co2_eq_emissions",
    "base_model",
]
IGNORED_FIELDS = ["model"]


@dataclass
class SpanMarkerModelCardData(CardData):
    """A dataclass storing data used in the model card.

    Args:
        language (Optional[Union[str, List[str]]]): The model language, either a string or a list,
            e.g. "en" or ["en", "de", "nl"]
        license: (Optional[str]): The license of the model, e.g. "apache-2.0", "mit"
            or "cc-by-nc-sa-4.0"
        model_name: (Optional[str]): The pretty name of the model, e.g. "SpanMarker with mBERT-base on CoNLL03".
            If not defined, uses encoder_name/encoder_id and dataset_name/dataset_id to generate a model name.
        model_id: (Optional[str]): The model ID when pushing the model to the Hub,
            e.g. "tomaarsen/span-marker-mbert-base-multinerd".
        encoder_name: (Optional[str]): The pretty name of the encoder, e.g. "mBERT-base".
        encoder_id: (Optional[str]): The model ID of the encoder, e.g. "bert-base-multilingual-cased".
        dataset_name: (Optional[str]): The pretty name of the dataset, e.g. "CoNLL03".
        dataset_id: (Optional[str]): The dataset ID of the dataset, e.g. "tner/bionlp2004".
        dataset_revision: (Optional[str]): The dataset revision/commit that was for training/evaluation.

    Note:

        Install ``nltk`` to detokenize the examples used in the model card, i.e. attach punctuation and brackets.
        Additionally, ``codecarbon`` can be installed to automatically track carbon emission usage.

    Example::

        >>> model = SpanMarkerModel.from_pretrained(
        ...     "bert-base-uncased",
        ...     labels=["O", "B-DNA", "I-DNA", "B-protein", ...],
        ...     # SpanMarker hyperparameters:
        ...     model_max_length=256,
        ...     marker_max_length=128,
        ...     entity_max_length=8,
        ...     # Model card variables
        ...     model_card_data=SpanMarkerModelCardData(
        ...         model_id="tomaarsen/span-marker-bbu-bionlp",
        ...         encoder_id="bert-base-uncased",
        ...         dataset_name="BioNLP2004,
        ...         dataset_id="tner/bionlp2004",
        ...         license="apache-2.0",
        ...         language="en",
        ...     ),
        ... )
    """

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
    model_name: Optional[str] = None
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
    predict_example: Optional[str] = field(default=None, init=False)
    label_example_list: List[Dict[str, str]] = field(default_factory=list, init=False)
    tokenizer_warning: bool = field(default=False, init=False)
    train_set_metrics_list: List[Dict[str, str]] = field(default_factory=list, init=False)
    code_carbon_callback: Optional[CodeCarbonCallback] = field(default=None, init=False)

    # Computed once, always unchanged
    pipeline_tag: str = field(default="token-classification", init=False)
    library_name: str = field(default="span-marker", init=False)
    version: Dict[str, str] = field(
        default_factory=lambda: {
            "python": python_version(),
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
    model: Optional["SpanMarkerModel"] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # We don't want to save "ignore_metadata_errors" in our Model Card
        if self.dataset_id:
            if is_on_huggingface(self.dataset_id, is_model=False):
                if self.language is None:
                    # if languages are not set, try to determine the language from the dataset on the Hub
                    try:
                        info = dataset_info(self.dataset_id)
                    except:
                        pass
                    else:
                        if info.cardData:
                            self.language = info.cardData.get("language", self.language)
            else:
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

    def set_widget_examples(self, dataset: Dataset) -> None:
        # If NLTK is installed, use its detokenization. Otherwise, join by spaces.
        try:
            from nltk.tokenize.treebank import TreebankWordDetokenizer

            detokenize = TreebankWordDetokenizer().detokenize

            def map_detokenize(tokens) -> Dict[str, str]:
                return {"text": detokenize(tokens)}

        except ImportError:

            def map_detokenize(tokens) -> Dict[str, str]:
                return {"text": " ".join(tokens)}

        # Out of `sample_subset_size=100` random samples, select `example_count=5` good examples
        # based on the number of unique entity classes.
        # The shortest example is used in the inference example
        sample_subset_size = 100
        example_count = 5
        if len(dataset) > sample_subset_size:
            example_dataset = dataset.select(random.sample(range(len(dataset)), k=sample_subset_size))
        else:
            example_dataset = dataset

        def count_entities(sample: Dict[str, Any]) -> Dict[str, int]:
            unique_count = {reduced_label_id for reduced_label_id, _, _ in sample["ner_tags"]}
            return {"unique_entity_count": len(unique_count)}

        example_dataset = (
            example_dataset.map(count_entities)
            .sort(("unique_entity_count", "entity_count"), reverse=True)
            .select(range(min(len(example_dataset), example_count)))
            .map(map_detokenize, input_columns="tokens")
        )
        self.widget = [{"text": sample["text"]} for sample in example_dataset]

        shortest_example = example_dataset.sort("word_count")[0]["text"]
        self.predict_example = shortest_example

    def set_train_set_metrics(self, dataset: Dataset) -> None:
        self.train_set_metrics_list = [
            {
                "Training set": "Sentence length",
                "Min": min(dataset["word_count"]),
                "Median": sum(dataset["word_count"]) / len(dataset),
                "Max": max(dataset["word_count"]),
            },
            {
                "Training set": "Entities per sentence",
                "Min": min(dataset["entity_count"]),
                "Median": sum(dataset["entity_count"]) / len(dataset),
                "Max": max(dataset["entity_count"]),
            },
        ]

    def set_label_examples(self, dataset: Dataset, id2label: Dict[int, str], outside_id: int) -> None:
        num_examples_per_label = 3
        examples = {label: set() for label_id, label in id2label.items() if label_id != outside_id}
        unfinished_entity_ids = set(id2label.keys()) - {outside_id}
        for sample in dataset:
            for entity_id, start, end in sample["ner_tags"]:
                if entity_id in unfinished_entity_ids:
                    entity = id2label[entity_id]
                    example = " ".join(sample["tokens"][start:end])
                    examples[entity].add(f'"{example}"')
                    if len(examples[entity]) >= num_examples_per_label:
                        unfinished_entity_ids.remove(entity_id)
            if not unfinished_entity_ids:
                break
        self.label_example_list = [
            {"Label": label, "Examples": ", ".join(example_set)} for label, example_set in examples.items()
        ]

    def infer_dataset_id(self, dataset: Dataset) -> None:
        def subtuple_finder(tuple: Tuple[str], subtuple: Tuple[str]) -> int:
            for i, element in enumerate(tuple):
                if element == subtuple[0] and tuple[i : i + len(subtuple)] == subtuple:
                    return i
            return -1

        def normalize(dataset_id: str) -> str:
            for token in "/\\_-":
                dataset_id = dataset_id.replace(token, "")
            return dataset_id.lower()

        if (cache_files := dataset.cache_files) and "filename" in cache_files[0]:
            cache_path_parts = Path(cache_files[0]["filename"]).parts
            # Check if the cachefile is under "huggingface/datasets"
            subtuple = ("huggingface", "datasets")
            index = subtuple_finder(cache_path_parts, subtuple)
            if index == -1:
                return

            # Get the folder after "huggingface/datasets"
            cache_dataset_name = cache_path_parts[index + len(subtuple)]
            # If the dataset has an author:
            if "___" in cache_dataset_name:
                author, dataset_name = cache_dataset_name.split("___")
            else:
                author = None
                dataset_name = cache_dataset_name

            # Make sure the normalized dataset IDs match
            dataset_list = [
                dataset
                for dataset in list_datasets(author=author, dataset_name=dataset_name)
                if normalize(dataset.id) == normalize(cache_dataset_name)
            ]
            # If there's only one match, get the ID from it
            if len(dataset_list) == 1:
                self.dataset_id = dataset_list[0].id

    def register_model(self, model: "SpanMarkerModel") -> None:
        self.model = model

        if self.encoder_id is None:
            encoder_id_or_path = self.model.config.get("_name_or_path")
            if not os.path.exists(encoder_id_or_path):
                self.encoder_id = encoder_id_or_path

        if not self.model_name:
            if self.encoder_id:
                self.model_name = f"SpanMarker with {self.encoder_name or self.encoder_id}"
                if self.dataset_name or self.dataset_id:
                    self.model_name += f" on {self.dataset_name or self.dataset_id}"
            else:
                self.model_name = "SpanMarker"

    def to_dict(self) -> Dict[str, Any]:
        super_dict = {field.name: getattr(self, field.name) for field in fields(self)}

        # Compute required formats from the raw data
        if self.eval_results_dict:
            dataset_split = list(self.eval_results_dict.keys())[0].split("_")[0]
            dataset_id = self.dataset_id or "unknown"
            dataset_name = self.dataset_name or "Unknown"
            eval_results = [
                EvalResult(
                    task_type="token-classification",
                    dataset_type=dataset_id,
                    dataset_name=dataset_name,
                    metric_type="f1",
                    metric_value=self.eval_results_dict[f"{dataset_split}_overall_f1"],
                    task_name="Named Entity Recognition",
                    dataset_split=dataset_split,
                    dataset_revision=self.dataset_revision,
                    metric_name="F1",
                ),
                EvalResult(
                    task_type="token-classification",
                    dataset_type=dataset_id,
                    dataset_name=dataset_name,
                    metric_type="precision",
                    metric_value=self.eval_results_dict[f"{dataset_split}_overall_precision"],
                    task_name="Named Entity Recognition",
                    dataset_split=dataset_split,
                    dataset_revision=self.dataset_revision,
                    metric_name="Precision",
                ),
                EvalResult(
                    task_type="token-classification",
                    dataset_type=dataset_id,
                    dataset_name=dataset_name,
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
        # Replace |:---:| with |:---| for left alignment
        super_dict["label_examples"] = make_markdown_table(self.label_example_list).replace("-:|", "--|")
        super_dict["train_set_metrics"] = make_markdown_table(self.train_set_metrics_list).replace("-:|", "--|")
        super_dict["metrics_table"] = make_markdown_table(self.metric_lines).replace("-:|", "--|")
        if self.code_carbon_callback and self.code_carbon_callback.tracker:
            emissions_data = self.code_carbon_callback.tracker._prepare_emissions_data()
            super_dict["co2_eq_emissions"] = {
                # * 1000 to convert kg to g
                "emissions": float(emissions_data.emissions) * 1000,
                "source": "codecarbon",
                "training_type": "fine-tuning",
                "on_cloud": emissions_data.on_cloud == "Y",
                "cpu_model": emissions_data.cpu_model,
                "ram_total_size": emissions_data.ram_total_size,
                "hours_used": round(emissions_data.duration / 3600, 3),
            }
            if emissions_data.gpu_model:
                super_dict["co2_eq_emissions"]["hardware_used"] = emissions_data.gpu_model
        if self.dataset_id:
            super_dict["datasets"] = [self.dataset_id]
        if self.encoder_id:
            super_dict["base_model"] = self.encoder_id
        super_dict["model_max_length"] = self.model.tokenizer.model_max_length

        for key in IGNORED_FIELDS:
            super_dict.pop(key, None)
        return {
            **self.model.config.to_dict(),
            **super_dict,
        }

    def to_yaml(self, line_break=None) -> str:
        return yaml_dump(
            {key: value for key, value in self.to_dict().items() if key in YAML_FIELDS and value is not None},
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
