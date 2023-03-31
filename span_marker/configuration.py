from collections import defaultdict
from typing import Any, Dict, Iterable, Optional, Set, Union

from transformers import PretrainedConfig


class SpanMarkerConfig(PretrainedConfig):
    model_type: str = "span-marker"

    def __init__(
        self,
        encoder_config: Optional[Dict[str, Any]] = None,
        model_max_length: Optional[int] = None,
        marker_max_length: int = 256,
        entity_max_length: int = 16,
        **kwargs,
    ) -> None:
        self.encoder = encoder_config
        self.model_max_length = model_max_length
        self.model_max_length_default = 512
        self.marker_max_length = marker_max_length
        self.entity_max_length = entity_max_length
        super().__init__(**kwargs)
        # These are automatically set by super().__init__, but we want to rely on
        # the encoder configs instead if we can, so we delete them.
        del self.id2label
        del self.label2id

        if encoder_config is None:
            return

        # If the id2label of the encoder is not overridden
        if self.id2label == {0: "LABEL_0", 1: "LABEL_1"}:
            raise ValueError(
                "Please provide a `labels` list to `SpanMarkerModel.from_pretrained()`, e.g.\n"
                ">>> SpanMarkerModel.from_pretrained(\n"
                '...     "bert-base-cased",\n'
                '...     labels=["O", "B-PER", "I-PER", "B-ORG", "I-ORG", ...]\n'
                "... )\n"
                "or\n"
                ">>> SpanMarkerModel.from_pretrained(\n"
                '...     "bert-base-cased",\n'
                '...     labels=["O", "PER", "ORG", "LOC", "MISC"]\n'
                "... )"
            )

        if self.id2label and "O" not in self.label2id:
            raise Exception("There must be an 'O' label.")

        # TODO: Consider converting this into several properties
        if self.are_labels_schemed():
            reduced_labels = {label[2:] for label in self.label2id.keys() if label != "O"}
            reduced_labels = ["O"] + sorted(reduced_labels)
            self.id2reduced_id = {
                _id: reduced_labels.index(label[2:] if label != "O" else label) for label, _id in self.label2id.items()
            }
            self.id2label = dict(enumerate(reduced_labels))
            self.label2id = {v: k for k, v in self.id2label.items()}
            self.outside_id = 0
        else:
            self.outside_id = self.label2id["O"]

    def __setattr__(self, name, value) -> None:
        """Whenever the vocab_size is updated, update it for both the SpanMarkerConfig and the
        underlying encoder config.
        """
        if name == "vocab_size":
            self.encoder[name] = value
        return super().__setattr__(name, value)

    def __getattribute__(self, key: str):
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
