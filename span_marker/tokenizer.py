import itertools
import os
import warnings
from typing import Any, Dict, Iterator, List, Tuple, Union

from transformers import AutoTokenizer, PreTrainedTokenizer

from span_marker.configuration import SpanMarkerConfig


class SpanMarkerTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizer, config: SpanMarkerConfig, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.config = config

        tokenizer.add_tokens(["<start>", "<end>"], special_tokens=True)
        self.start_marker_id, self.end_marker_id = self.tokenizer.convert_tokens_to_ids(["<start>", "<end>"])

        if self.tokenizer.model_max_length > 1e29 and self.config.model_max_length is None:
            warnings.warn(
                f"Base {self.tokenizer.__class__.__name__} nor {self.config.__class__.__name__} specify"
                f" `model_max_length`: defaulting to {self.model_max_length_default} tokens."
            )
        self.model_max_length = min(
            self.tokenizer.model_max_length, self.config.model_max_length or self.config.model_max_length_default
        )

    def get_all_valid_spans(self, num_words: int, entity_max_length: int) -> Iterator[Tuple[int, int]]:
        for start_idx in range(num_words):
            for end_idx in range(start_idx + 1, min(num_words + 1, start_idx + 1 + entity_max_length)):
                yield (start_idx, end_idx)

    def get_all_valid_spans_and_labels(
        self, num_words: int, span_to_label: Dict[Tuple[int, int], int], entity_max_length: int, outside_id: int
    ) -> Iterator[Tuple[Tuple[int, int], int]]:
        for span in self.get_all_valid_spans(num_words, entity_max_length):
            yield span, span_to_label.get(span, outside_id)

    def __getattribute__(self, key: str) -> Any:
        try:
            return super().__getattribute__(key)
        except AttributeError:
            return super().__getattribute__("tokenizer").__getattribute__(key)

    def __call__(
        self, inputs, labels=None, return_num_words: bool = False, return_batch_encoding=False, **kwargs
    ) -> Dict[str, List]:
        # TODO: Increase robustness of this
        is_split_into_words = True
        if isinstance(inputs, str) or (inputs and " " in inputs[0]):
            is_split_into_words = False

        batch_encoding = self.tokenizer(
            inputs,
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
            word_ids = itertools.takewhile(lambda word_id: word_id is not None, batch_encoding.word_ids(sample_idx)[1:])
            num_words = max(word_ids) + 1
            if self.tokenizer.pad_token_id in input_ids:
                num_tokens = list(input_ids).index(self.tokenizer.pad_token_id)
            else:
                num_tokens = len(input_ids)
            if labels:
                span_to_label = {(start_idx, end_idx): label for label, start_idx, end_idx in labels[sample_idx]}
                spans, span_labels = zip(
                    *list(
                        self.get_all_valid_spans_and_labels(
                            num_words, span_to_label, self.config.entity_max_length, self.config.outside_id
                        )
                    )
                )
            else:
                spans = list(self.get_all_valid_spans(num_words, self.config.entity_max_length))

            for group_start_idx in range(0, len(spans), self.config.marker_max_length):
                group_spans = spans[group_start_idx : group_start_idx + self.config.marker_max_length]
                group_num_spans = len(group_spans)

                start_position_ids, end_position_ids = [], []
                for start_word_i, end_word_i in group_spans:
                    start_token_span = batch_encoding.word_to_tokens(sample_idx, word_index=start_word_i)
                    start_position_ids.append(start_token_span.start if start_token_span else 0)

                    end_token_span = batch_encoding.word_to_tokens(sample_idx, word_index=end_word_i - 1)
                    end_position_ids.append(end_token_span.end - 1 if end_token_span else 0)

                all_input_ids.append(input_ids[:num_tokens])
                all_num_spans.append(group_num_spans)
                all_start_position_ids.append(start_position_ids)
                all_end_position_ids.append(end_position_ids)

                if labels:
                    group_labels = span_labels[group_start_idx : group_start_idx + self.config.marker_max_length]
                    all_labels.append(group_labels)

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
