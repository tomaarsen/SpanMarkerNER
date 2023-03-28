from collections import defaultdict
from typing import Any, Dict, Optional, Set

from transformers import PretrainedConfig


class SpanMarkerConfig(PretrainedConfig):
    model_type: str = "span-marker"

    def __init__(
        self, encoder_config: Optional[Dict[str, Any]] = None, max_marker_length=256, max_entity_length=16, **kwargs
    ) -> None:
        # if encoder_config is None:
        #     raise Exception("`encoder_config` must be provided.")
        self.encoder = encoder_config
        self.max_marker_length = max_marker_length
        self.max_entity_length = max_entity_length
        super().__init__(**kwargs)
        if encoder_config is None:
            return

        # These are automatically set, but we want to rely on the encoder configs instead if we can,
        # so we delete them.
        del self.id2label
        del self.label2id

        if self.id2label and "O" not in self.label2id:
            raise Exception("There must be an O label.")

        # TODO: Consider converting this into several properties
        if self.are_labels_schemed():
            reduced_labels = {label[2:] for label in self.label2id.keys() if label != "O"}
            reduced_labels = ["O"] + sorted(reduced_labels)
            self.id2reduced_id = {
                _id: reduced_labels.index(label[2:] if label != "O" else label) for label, _id in self.label2id.items()
            }
            self.id2label = dict(enumerate(reduced_labels))
            self.label2id = {v: k for k, v in self.id2label.items()}
            self.outside_id = self.id2reduced_id[self.label2id["O"]]  # <- always 0
        else:
            self.id2reduced_id = None
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
