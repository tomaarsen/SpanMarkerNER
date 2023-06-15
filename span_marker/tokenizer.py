import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer

from span_marker.configuration import SpanMarkerConfig

logger = logging.getLogger(__name__)


@dataclass
class EntityTracker:
    """
    For giving a warning about what percentage of entities are ignored/skipped.

    Example::

        This SpanMarker model won't be able to predict 5.930931% of all annotated entities in the evaluation dataset.
        This is caused by the SpanMarkerModel maximum entity length of 6 words and the maximum model input length of 64 tokens.
        These are the frequencies of the missed entities due to maximum entity length out of 1332 total entities:
        - 7 missed entities with 7 words (0.525526%)
        - 2 missed entities with 8 words (0.150150%)
        - 2 missed entities with 9 words (0.150150%)
        - 2 missed entities with 13 words (0.150150%)
        Additionally, a total of 66 (4.954955%) entities were missed due to the maximum input length.
    """

    entity_max_length: int
    model_max_length: int
    split: str = "train"  # or "evaluation" or "test"
    total_num_entities: int = 0
    skipped_entities: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    enabled: bool = False

    def __call__(self, split: Optional[str] = None) -> None:
        """Update the current split, which affects the warning message.

        Example:

            with tokenizer.entity_tracker(split=dataset_name):
                dataset = dataset.map(
                    tokenizer,
                    ...
                )

        Args:
            split (Optional[str]): The new split string, either "train", "evaluation" or "test". Defaults to None.

        Returns:
            Self: The EntityTracker instance.
        """
        if split:
            self.split = split
        return self

    def __enter__(self) -> None:
        """Start tracking (ignored) entities on enter."""
        self.enabled = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Trigger the ignored entities warning on exit."""
        if not self.skipped_entities:
            return self.reset()

        entity_max_length_missed_freq = sorted(
            [(key, value) for key, value in self.skipped_entities.items() if key > self.entity_max_length],
            key=lambda x: x[0],
        )
        model_max_length_missed_total = sum(
            value for key, value in self.skipped_entities.items() if key <= self.entity_max_length
        )
        total_num_missed_entities = sum(self.skipped_entities.values())
        if self.split == "train":
            message = "This SpanMarker model will ignore"
        else:
            message = "This SpanMarker model won't be able to predict"
        message += (
            f" {total_num_missed_entities/self.total_num_entities:%} of all annotated entities in the {self.split}"
            " dataset. This is caused by the SpanMarkerModel "
        )
        if entity_max_length_missed_freq:
            message += (
                f"maximum entity length of {self.entity_max_length} word{'s' if self.entity_max_length > 1 else ''}"
            )
            if model_max_length_missed_total:
                message += " and the "
        if model_max_length_missed_total:
            message += (
                f"maximum model input length of {self.model_max_length} token{'s' if self.model_max_length > 1 else ''}"
            )
        message += "."
        if entity_max_length_missed_freq:
            message += f"\nThese are the frequencies of the missed entities due to maximum entity length out of {self.total_num_entities} total entities:\n"
            message += "\n".join(
                [
                    f"- {freq} missed entities with {length} word{'s' if length > 1 else ''}"
                    f" ({freq / self.total_num_entities:%})"
                    for length, freq in entity_max_length_missed_freq
                ]
            )
        if model_max_length_missed_total:
            if entity_max_length_missed_freq:
                message += "\nAdditionally, a "
            else:
                message += "\nA "
            message += (
                f"total of {model_max_length_missed_total} ({model_max_length_missed_total / self.total_num_entities:%})"
                " entities were missed due to the maximum input length."
            )
        logger.warning(message)
        self.reset()

    def add(self, num_entities: int) -> None:
        """Add to the counter of total number of entities.

        Args:
            num_entities (int): How many entities to increment by.
        """
        self.total_num_entities += num_entities

    def missed(self, length: int) -> None:
        """Add to the counter of missed/ignored/skipped entities.

        Args:
            length (int): How many entities were missed.
        """
        self.skipped_entities[length] += 1

    def reset(self) -> None:
        """Reset to defaults, stops tracking."""
        self.total_num_entities = 0
        self.skipped_entities = defaultdict(int)
        self.enabled = False


class SpanMarkerTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizer, config: SpanMarkerConfig, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.config = config

        tokenizer.add_tokens(["<start>", "<end>"], special_tokens=True)
        self.start_marker_id, self.end_marker_id = self.tokenizer.convert_tokens_to_ids(["<start>", "<end>"])

        if self.tokenizer.model_max_length > 1e29 and self.config.model_max_length is None:
            logger.warning(
                f"The underlying {self.tokenizer.__class__.__name__!r} tokenizer nor {self.config.__class__.__name__!r}"
                f" specify `model_max_length`: defaulting to {self.config.model_max_length_default} tokens."
            )
        self.model_max_length = min(
            self.tokenizer.model_max_length, self.config.model_max_length or self.config.model_max_length_default
        )

        self.entity_tracker = EntityTracker(self.config.entity_max_length, self.model_max_length)

    def get_all_valid_spans(self, num_words: int, entity_max_length: int) -> Iterator[Tuple[int, int]]:
        for start_idx in range(num_words):
            for end_idx in range(start_idx + 1, min(num_words + 1, start_idx + 1 + entity_max_length)):
                yield (start_idx, end_idx)

    def get_all_valid_spans_and_labels(
        self, num_words: int, span_to_label: Dict[Tuple[int, int], int], entity_max_length: int, outside_id: int
    ) -> Iterator[Tuple[Tuple[int, int], int]]:
        for span in self.get_all_valid_spans(num_words, entity_max_length):
            yield span, span_to_label.pop(span, outside_id)

    def __getattribute__(self, key: str) -> Any:
        try:
            return super().__getattribute__(key)
        except AttributeError:
            return super().__getattribute__("tokenizer").__getattribute__(key)

    def __call__(
        self, batch: Dict[str, List[Any]], return_num_words: bool = False, return_batch_encoding=False, **kwargs
    ) -> Dict[str, List]:
        tokens = batch["tokens"]
        labels = batch.get("ner_tags", None)
        # TODO: Increase robustness of this
        is_split_into_words = True
        if isinstance(tokens, str) or (tokens and " " in tokens[0]):
            is_split_into_words = False

        batch_encoding = self.tokenizer(
            tokens,
            **kwargs,
            is_split_into_words=is_split_into_words,
            padding="max_length",
            truncation=True,
            max_length=self.model_max_length,
            return_tensors="pt",
        )

        all_input_ids = []
        all_num_spans = []
        all_start_position_ids = []
        all_end_position_ids = []
        all_labels = []
        all_num_words = []
        for sample_idx, input_ids in enumerate(batch_encoding["input_ids"]):
            num_words = int(np.nanmax(np.array(batch_encoding.word_ids(sample_idx), dtype=float))) + 1
            if self.tokenizer.pad_token_id in input_ids:
                num_tokens = list(input_ids).index(self.tokenizer.pad_token_id)
            else:
                num_tokens = len(input_ids)
            if labels:
                span_to_label = {(start_idx, end_idx): label for label, start_idx, end_idx in labels[sample_idx]}
                if self.entity_tracker.enabled:
                    self.entity_tracker.add(len(span_to_label))
                spans, span_labels = zip(
                    *list(
                        self.get_all_valid_spans_and_labels(
                            num_words, span_to_label, self.config.entity_max_length, self.config.outside_id
                        )
                    )
                )
                # `self.get_all_valid_spans_and_labels` popped `span_to_label`, so if it's non-empty, then that
                # entity was ignored, and we may want to track it for a useful warning
                if self.entity_tracker.enabled:
                    for start, end in span_to_label.keys():
                        self.entity_tracker.missed(end - start)
            else:
                spans = list(self.get_all_valid_spans(num_words, self.config.entity_max_length))

            start_position_ids, end_position_ids = [], []
            for start_word_i, end_word_i in spans:
                start_token_span = batch_encoding.word_to_tokens(sample_idx, word_index=start_word_i)
                # The if ... else 0 exists because of words like '\u2063'
                start_position_ids.append(start_token_span.start if start_token_span else 0)

                end_token_span = batch_encoding.word_to_tokens(sample_idx, word_index=end_word_i - 1)
                end_position_ids.append(end_token_span.end - 1 if end_token_span else 0)

            all_input_ids.append(input_ids[:num_tokens].tolist())
            all_num_spans.append(len(spans))
            all_start_position_ids.append(start_position_ids)
            all_end_position_ids.append(end_position_ids)

            if labels:
                all_labels.append(span_labels)

            if return_num_words:
                all_num_words.append(num_words)

        output = {
            "input_ids": all_input_ids,
            "num_spans": all_num_spans,
            "start_position_ids": all_start_position_ids,
            "end_position_ids": all_end_position_ids,
        }
        if labels:
            output["labels"] = all_labels
        if return_num_words:
            # Store the number of words, useful for computing the spans in the evaluation and model.predict() method
            output["num_words"] = all_num_words
        if return_batch_encoding:
            # Store the batch encoding, useful for converting word IDs to characters in the model.predict() method
            output["batch_encoding"] = batch_encoding
        return output

    def __len__(self) -> int:
        return len(self.tokenizer)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], *inputs, config=None, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, *inputs, **kwargs, add_prefix_space=True
        )
        return cls(tokenizer, config=config, **kwargs)
