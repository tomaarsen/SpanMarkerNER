import itertools
import os
from typing import Dict, Generator, Iterator, List, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, PreTrainedTokenizer

from span_marker.configuration import SpanMarkerConfig


class SpanMarkerTokenizer:
    # def __init__(self, model: SpanMarkerModel, tokenizer: PreTrainedTokenizer, **kwargs):
    def __init__(self, tokenizer: PreTrainedTokenizer, config=None, **kwargs) -> None:
        # super().__init__(**kwargs)
        # self.model = model
        self.tokenizer = tokenizer
        self.config = config

        tokenizer.add_tokens(["<start>", "<end>"], special_tokens=True)
        # tokenizer.add_special_tokens({"additional_special_tokens": ["<start>", "<end>"]})
        self.start_marker_id, self.end_marker_id = self.tokenizer.convert_tokens_to_ids(["<start>", "<end>"])
        # self.start_marker_id, self.end_marker_id = self.tokenizer.convert_tokens_to_ids(['madeupword0000', 'madeupword0001'])

        # TODO: This could be done more cleverly. Perhaps I can just subclass PreTrainedTokenizerFast?
        # I'm concerned about .from_pretrained not initializing a SpanMarkerTokenizer though.
        self.pad = tokenizer.pad
        self.save_pretrained = tokenizer.save_pretrained
        self.decode = tokenizer.decode
        self.model_max_length = self.tokenizer.model_max_length
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def get_all_valid_spans(self, num_words: int, max_entity_length: int) -> Iterator[Tuple[int, int]]:
        for start_idx in range(num_words):
            for end_idx in range(start_idx + 1, min(num_words + 1, start_idx + 1 + max_entity_length)):
                yield (start_idx, end_idx)

    def get_all_valid_spans_and_labels(
        self, num_words: int, label_ids_dict: Dict[Tuple[int, int], int], max_entity_length: int, outside_id: int
    ) -> Iterator[Tuple[Tuple[int, int], int]]:
        for span in self.get_all_valid_spans(num_words, max_entity_length):
            yield span, label_ids_dict.get(span, outside_id)

    def __call__(
        self, inputs, config: SpanMarkerConfig = None, labels=None, return_num_words: bool = False, **kwargs
    ) -> Dict[str, List]:
        config = config or self.config
        if config is None:
            raise Exception(
                "Please provide `SpanMarkerTokenizer` with a `SpanMarkerConfig` instance via the `config` keyword argument."
            )
        self.config = config

        # TODO: Increase robustness of this
        # TODO: Ensure that inputs is a list of pretokenized words
        is_split_into_words = True
        if isinstance(inputs, str) or (inputs and " " in inputs[0]):
            is_split_into_words = False

        # TODO: one-by-one tokenization to create smaller input_ids?
        # input_ids are already shrunk later on
        kwargs["padding"] = "max_length"
        kwargs["return_tensors"] = "pt"
        # TODO: Undo this
        batch_encoding = self.tokenizer(
            inputs, **kwargs, is_split_into_words=is_split_into_words, max_length=256
        )  # <- TODO: Remove the hardcoding!

        all_input_ids = []
        all_num_spans = []
        all_start_position_ids = []
        all_end_position_ids = []
        all_labels = []
        all_num_words = []

        for sample_idx, input_ids in enumerate(batch_encoding["input_ids"]):
            word_ids = itertools.takewhile(lambda word_id: word_id is not None, batch_encoding.word_ids(sample_idx)[1:])
            num_words = max(word_ids) + 1
            num_tokens = list(input_ids).index(self.tokenizer.pad_token_id)
            if labels:
                label_ids_dict = {(start_idx, end_idx): label for label, start_idx, end_idx in labels[sample_idx]}
                spans, span_labels = zip(
                    *list(
                        self.get_all_valid_spans_and_labels(
                            num_words, label_ids_dict, config.max_entity_length, config.outside_id
                        )
                    )
                )
            else:
                spans = list(self.get_all_valid_spans(num_words, config.max_entity_length))

            for group_start_idx in range(0, len(spans), config.max_marker_length):
                group_spans = spans[group_start_idx : group_start_idx + config.max_marker_length]
                group_word_starts, group_word_ends = zip(*group_spans)
                group_num_spans = len(group_spans)

                start_position_ids = [
                    batch_encoding.word_to_tokens(sample_idx, word_index=word_i).start for word_i in group_word_starts
                ]
                end_position_ids = [
                    batch_encoding.word_to_tokens(sample_idx, word_index=word_i - 1).end - 1
                    for word_i in group_word_ends
                ]

                all_input_ids.append(input_ids[:num_tokens])
                all_num_spans.append(group_num_spans)
                all_start_position_ids.append(start_position_ids)
                all_end_position_ids.append(end_position_ids)

                if labels:
                    group_labels = span_labels[group_start_idx : group_start_idx + config.max_marker_length]
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
            output["num_words"] = all_num_words
        return output

    def __len__(self) -> int:
        return len(self.tokenizer)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], *inputs, config=None, **kwargs):
        # TODO: Consider subclassing an AutoTokenizer directly instead of loading one like this:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, *inputs, **kwargs, add_prefix_space=True
        )
        return cls(tokenizer, config=config, **kwargs)
