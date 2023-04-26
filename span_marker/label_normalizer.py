import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Tuple

from span_marker.configuration import SpanMarkerConfig

logger = logging.getLogger(__name__)

Entity = Tuple[int, int, int]
"""
Tuple of:

* Entity label
* Word start index
* Word end index
"""


class LabelNormalizer(ABC):
    """Class to convert NER training data into a common format used in the :class:`~span_marker.tokenizer.SpanMarkerTokenizer`.

    The common format involves 3-tuples with entity labels, word start indices and word end indices.
    """

    def __init__(self, config: SpanMarkerConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def __call__(self, tokens: List[str], ner_tags: List[int]) -> Dict[str, List[Any]]:
        raise NotImplementedError


class LabelNormalizerScheme(LabelNormalizer):
    def __init__(self, config: SpanMarkerConfig) -> None:
        super().__init__(config)
        self.label_ids_by_tag = self.config.group_label_ids_by_tag()
        self.start_ids = set()
        self.end_ids = set()

    def ner_tags_to_entities(self, ner_tags: List[int]) -> Iterator[Entity]:
        """Assumes a correct IOB or IOB2 annotation scheme"""
        start_idx = None
        reduced_label_id = None
        for idx, label_id in enumerate(ner_tags):
            # End of an entity
            if start_idx is not None and label_id in self.end_ids:
                yield (reduced_label_id, start_idx, idx)
                start_idx = None

            # Start of an entity
            if start_idx is None and label_id in self.start_ids:
                # compute the schemeless label ID
                reduced_label_id = self.config.id2reduced_id[label_id]
                start_idx = idx

        if start_idx is not None:
            yield (reduced_label_id, start_idx, idx + 1)

    def __call__(self, tokens: List[str], ner_tags: List[int]) -> Dict[str, List[Any]]:
        return {"tokens": tokens, "ner_tags": list(self.ner_tags_to_entities(ner_tags))}


class LabelNormalizerIOB(LabelNormalizerScheme):
    def __init__(self, config: SpanMarkerConfig) -> None:
        super().__init__(config)
        # Support for IOB2 and IOB, respectively:
        logger.info("Detected the IOB or IOB2 labeling scheme.")
        self.start_ids = self.label_ids_by_tag["B"] | self.label_ids_by_tag["I"]
        self.end_ids = self.label_ids_by_tag["B"] | self.label_ids_by_tag["O"]


class LabelNormalizerBIOES(LabelNormalizerScheme):
    def __init__(self, config: SpanMarkerConfig) -> None:
        super().__init__(config)
        logger.info("Detected the BIOES labeling scheme.")
        self.start_ids = self.label_ids_by_tag["B"] | self.label_ids_by_tag["S"]
        self.end_ids = self.label_ids_by_tag["B"] | self.label_ids_by_tag["O"] | self.label_ids_by_tag["S"]


class LabelNormalizerBILOU(LabelNormalizerScheme):
    def __init__(self, config: SpanMarkerConfig) -> None:
        super().__init__(config)
        logger.info("Detected the BILOU labeling scheme.")
        self.start_ids = self.label_ids_by_tag["B"] | self.label_ids_by_tag["U"]
        self.end_ids = self.label_ids_by_tag["B"] | self.label_ids_by_tag["O"] | self.label_ids_by_tag["U"]


class LabelNormalizerNoScheme(LabelNormalizer):
    def __init__(self, config: SpanMarkerConfig) -> None:
        super().__init__(config)
        logger.info("No labeling scheme detected: all label IDs belong to individual entity classes.")

    def ner_tags_to_entities(self, ner_tags: List[int]) -> Iterator[Entity]:
        start_idx = None
        entity_label_id = None
        for idx, label_id in enumerate(ner_tags):
            # End of an entity
            if start_idx is not None and label_id != entity_label_id:
                yield (entity_label_id, start_idx, idx)
                start_idx = None

            # Start of an entity
            if start_idx is None and label_id != self.config.outside_id:
                entity_label_id = label_id
                start_idx = idx

        if start_idx is not None:
            yield (entity_label_id, start_idx, idx + 1)

    def __call__(self, tokens: List[str], ner_tags: List[int]) -> Dict[str, List[Any]]:
        return {"tokens": tokens, "ner_tags": list(self.ner_tags_to_entities(ner_tags))}


class AutoLabelNormalizer:
    """Factory class to return the correct LabelNormalizer subclass."""

    @staticmethod
    def from_config(config: SpanMarkerConfig) -> LabelNormalizer:
        if not config.are_labels_schemed():
            return LabelNormalizerNoScheme(config)

        tags = config.get_scheme_tags()
        if tags == set("BIO"):
            return LabelNormalizerIOB(config)
        if tags == set("BIOES"):
            return LabelNormalizerBIOES(config)
        if tags == set("BILOU"):
            return LabelNormalizerBILOU(config)
        raise ValueError(
            "Data labeling scheme not recognized. Expected either IOB, IOB2, BIOES, BILOU "
            "or no scheme (i.e. one label ID per class, no B-, I- label prefixes, etc.)"
        )
