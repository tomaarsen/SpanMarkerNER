from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.nn import functional as F

from span_marker.tokenizer import SpanMarkerTokenizer


@dataclass
class SpanMarkerDataCollator:
    """
    Data Collator class responsible for converting the minimal outputs from the tokenizer into
    complete and meaningful inputs to the model. In particular, the `input_ids` from the tokenizer
    features are padded, and the correct amount of start and end markers (with padding) are added.

    Furthermore, the position IDs are generated for the input IDs, and `start_position_ids` and
    `end_position_ids` are used alongside some padding to create a full position ID vector.

    Lastly, the attention matrix is computed.

    The expected usage is something like:

    >>> collator = SpanMarkerDataCollator(...)
    >>> tokenized = tokenizer(...)
    >>> batch = collator(tokenized)
    >>> output = model(**batch)
    """

    tokenizer: SpanMarkerTokenizer
    marker_max_length: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Convert the minimal tokenizer outputs into inputs ready for `model.forward`.

        Args:
            features (List[Dict[str, Any]]): A list of dictionaries, one element per sample in the batch.
                The dictionaries contain the following keys:

                * `input_ids`: The non-padded input IDs.
                * `num_spans`: The number of spans that should be encoded in each sample.
                * `start_position_ids`: The position IDs of the start markers in the sample.
                * `end_position_ids`: The position IDs of the end markers in the sample.
                * `labels` (optional): The labels corresponding to each of the spans in the sample.
                * `num_words` (optional): The number of words in the input sample.
                    Required for some evaluation metrics.

        Returns:
            Dict[str, torch.Tensor]: Batch dictionary ready to be fed into :meth:`~SpanMarkerModel.forward`.
        """
        total_size = self.tokenizer.model_max_length + 2 * self.marker_max_length
        start_marker_idx = self.tokenizer.model_max_length
        end_marker_idx = self.tokenizer.model_max_length + self.marker_max_length
        batch = defaultdict(list)
        num_words = []
        for sample in features:
            input_ids = sample["input_ids"]
            num_spans = sample["num_spans"]
            num_tokens = len(input_ids)

            # Prepare input_ids by padding and adding start and end markers
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids, dtype=torch.int)
            else:
                input_ids.to(torch.int)
            input_ids = F.pad(input_ids, (0, total_size - len(input_ids)), value=self.tokenizer.pad_token_id)
            input_ids[start_marker_idx : start_marker_idx + num_spans] = self.tokenizer.start_marker_id
            input_ids[end_marker_idx : end_marker_idx + num_spans] = self.tokenizer.end_marker_id
            batch["input_ids"].append(input_ids)

            # Prepare position IDs
            position_ids = torch.arange(num_tokens, dtype=torch.int) + 2
            position_ids = F.pad(position_ids, (0, total_size - len(position_ids)), value=0)
            position_ids[num_tokens]=1
            position_ids[start_marker_idx : start_marker_idx + num_spans] = (
                torch.tensor(sample["start_position_ids"]) + 2
            )
            position_ids[end_marker_idx : end_marker_idx + num_spans] = torch.tensor(sample["end_position_ids"]) + 2
            batch["position_ids"].append(position_ids)

            # Prepare attention mask matrix
            attention_mask = torch.zeros((total_size, total_size), dtype=torch.bool)
            # text tokens self-attention
            attention_mask[:num_tokens, :num_tokens] = 1
            # let markers attend text tokens
            attention_mask[start_marker_idx : start_marker_idx + num_spans, :num_tokens] = 1
            attention_mask[end_marker_idx : end_marker_idx + num_spans, :num_tokens] = 1
            # self-attentions of start/end markers
            start_index_list = list(range(start_marker_idx, start_marker_idx + num_spans))
            end_index_list = list(range(end_marker_idx, end_marker_idx + num_spans))
            attention_mask[start_index_list, start_index_list] = 1
            attention_mask[start_index_list, end_index_list] = 1
            attention_mask[end_index_list, start_index_list] = 1
            attention_mask[end_index_list, end_index_list] = 1
            batch["attention_mask"].append(attention_mask)

            if "labels" in sample:
                labels = torch.tensor(sample["labels"])
                labels = F.pad(labels, (0, self.marker_max_length - len(labels)), value=-100)
                batch["labels"].append(labels)

            if "num_words" in sample:
                num_words.append(sample["num_words"])

        batch = {key: torch.stack(value) for key, value in batch.items()}
        # Used for evaluation, does not need to be padded/stacked
        if num_words:
            batch["num_words"] = torch.tensor(num_words)

        return batch
