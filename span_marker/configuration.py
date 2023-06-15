from collections import defaultdict
from typing import Any, Dict, Iterable, Optional, Set, Union

from transformers import PretrainedConfig


class SpanMarkerConfig(PretrainedConfig):
    """Configuration class for SpanMarkerModel instances.

    Args:
        encoder_config (`Optional[Dict[str, Any]]`): The configuration dictionary for the
            underlying encoder used by the SpanMarkerModel instance. Defaults to None.
        model_max_length (`Optional[int]`): The total number of tokens that can be processed before
            truncation. If None, the tokenizer its `model_max_length` is used, and if that value is
            not defined, it becomes 512 instead. Defaults to None.
        marker_max_length (`int`): The maximum length for each of the span markers. A value of 128
            means that each training and inferencing sample contains a maximum of 128 start markers
            and 128 end markers, for a total of 256 markers per sample. Defaults to 128.
        entity_max_length (`int`): The maximum length of an entity span in terms of words.
            Defaults to 8.
        max_prev_context (`Optional[int]`): The maximum number of previous sentences to include as
            context. If `None`, the maximum amount that fits in `model_max_length` is chosen.
            Defaults to `None`.
        max_next_context (`Optional[int]`): The maximum number of next sentences to include as
            context. If `None`, the maximum amount that fits in `model_max_length` is chosen.
            Defaults to `None`.

    Example::

        # These configuration settings are provided via kwargs to `SpanMarkerModel.from_pretrained`:
        model = SpanMarkerModel.from_pretrained(
            "bert-base-cased",
            labels=labels,
            model_max_length=256,
            marker_max_length=128,
            entity_max_length=8,
        )

    Raises:
        ValueError: If the labels provided to :meth:`~span_marker.modeling.SpanMarkerModel.from_pretrained` do not
            contain the required `"O"` label.
    """

    model_type: str = "span-marker"
    is_composition = True

    def __init__(
        self,
        encoder_config: Optional[Dict[str, Any]] = None,
        model_max_length: Optional[int] = None,
        marker_max_length: int = 128,
        entity_max_length: int = 8,
        max_prev_context: Optional[int] = None,
        max_next_context: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.encoder = encoder_config
        self.model_max_length = model_max_length
        self.model_max_length_default = 512
        self.marker_max_length = marker_max_length
        self.entity_max_length = entity_max_length
        self.max_prev_context = max_prev_context
        self.max_next_context = max_next_context
        self.trained_with_document_context = False
        self.span_marker_version = kwargs.pop("span_marker_version", None)
        super().__init__(**kwargs)

        # label2id and id2label are automatically set by super().__init__, but we want to rely on
        # the encoder configs instead if we can, so we delete them under two conditions
        # 1. if they're the default ({0: "LABEL_0", 1: "LABEL_1"})
        # 2. if they're identical to the encoder label2id
        span_marker_label2id = super().__getattribute__("label2id")
        if span_marker_label2id == {"LABEL_0": 0, "LABEL_1": 1} or span_marker_label2id == self.encoder.get("label2id"):
            del self.id2label
            del self.label2id

        # We need the "O" label for label normalization, etc.
        if self.label2id and "O" not in self.label2id:
            raise ValueError("There must be an 'O' label in the list of `labels`.")

        # Keys are always strings in JSON so convert ids to int here.
        self.encoder["id2label"] = {int(label_id): label for label_id, label in self.encoder["id2label"].items()}
        if hasattr(self, "id2reduced_id"):
            self.id2reduced_id = {int(label_id): reduced_id for label_id, reduced_id in self.id2reduced_id.items()}
        elif self.are_labels_schemed():
            reduced_labels = {label[2:] for label in self.label2id.keys() if label != "O"}
            reduced_labels = ["O"] + sorted(reduced_labels)
            self.id2reduced_id = {
                _id: reduced_labels.index(label[2:] if label != "O" else label) for label, _id in self.label2id.items()
            }
            self.id2label = dict(enumerate(reduced_labels))
            self.label2id = {v: k for k, v in self.id2label.items()}

    @property
    def outside_id(self) -> None:
        return self.label2id["O"]

    def __setattr__(self, name, value) -> None:
        """Whenever the vocab_size is updated, update it for both the SpanMarkerConfig and the
        underlying encoder config.
        """
        if name == "vocab_size":
            self.encoder[name] = value
        # `outside_id` is now a property instead.
        if name == "outside_id":
            return
        return super().__setattr__(name, value)

    def __getattribute__(self, key: str) -> Any:
        try:
            return super().__getattribute__(key)
        except AttributeError as e:
            try:
                return super().__getattribute__("encoder")[key]
            except KeyError:
                raise e

    def are_labels_schemed(self) -> bool:
        """True if all labels are strings matching one of the two following rules:

        * `label == "O"`
        * `label[0] in "BIESLU"` and `label[1] == "-"`, e.g. in `"I-LOC"`

        We ensure that the first index is in `"BIELSU"` because of these definitions:
        * `"B"` for `"begin"`
        * `"I"` for `"in"`
        * `"E"` for `"end"`
        * `"L"` for `"last"`
        * `"S"` for `"singular"`
        * `"U"` for `"unit"`

        Args:
            id2label (Dict[int, str]): Dictionary of label ids to label strings.

        Returns:
            bool: True if it seems like a labeling scheme is used.
        """
        return self.encoder["id2label"] and all(
            label == "O" or (len(label) > 2 and label[0] in "BIELSU" and label[1] == "-")
            for label in self.encoder["id2label"].values()
        )

    def get_scheme_tags(self) -> Set[str]:
        return set(label[0] for label in self.encoder["id2label"].values())

    def group_label_ids_by_tag(self) -> Dict[str, Set]:
        grouped = defaultdict(set)
        for label_id, label in self.encoder["id2label"].items():
            grouped[label[0]].add(label_id)
        return dict(grouped)

    def get(self, options: Union[str, Iterable[str]], default: Any = None) -> Any:
        if isinstance(options, str):
            options = [options]
        for option in options:
            try:
                return self.__getattribute__(option)
            except AttributeError:
                pass
        return default
