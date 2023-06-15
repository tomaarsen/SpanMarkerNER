from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import TokenClassifierOutput


@dataclass
class SpanMarkerOutput(TokenClassifierOutput):
    """
    Class for outputs of :class:`~span_marker.modeling.SpanMarkerModel`.

    Args:
        loss (Optional[torch.FloatTensor]):
            Classification loss of shape ``(1,)``, returned when ``labels`` is provided.
        logits (torch.FloatTensor):
            Classification scores before softmax with shape ``(batch_size, sequence_length, config.num_labels)``.
        hidden_states (Optional[Tuple[torch.FloatTensor]]):
            Tuple of :class:`~torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape ``(batch_size, sequence_length, hidden_size)``.
            Returned when ``config.output_hidden_states=True``.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (Optional[Tuple[torch.FloatTensor]]):
            Tuple of :class:`~torch.FloatTensor` (one for each layer) of shape ``(batch_size, num_heads, sequence_length,
            sequence_length)``. Returned when ``config.output_attentions=True``.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        num_words (Optional[~torch.Tensor]):
            A vector with shape ``(batch_size,)`` that tracks how many words were in the input of each sample in the batch.
            Required for evaluation purposes.
        document_ids (Optional[~torch.Tensor]):
            A vector with shape ``(batch_size,)`` that tracks the document the input text belongs to.
        sentence_ids (Optional[~torch.Tensor]):
            A vector with shape ``(batch_size,)`` that tracks the sentence in the document that the input text belongs to.
    """

    num_marker_pairs: Optional[torch.Tensor] = None
    num_words: Optional[torch.Tensor] = None
    document_ids: Optional[torch.Tensor] = None
    sentence_ids: Optional[torch.Tensor] = None
