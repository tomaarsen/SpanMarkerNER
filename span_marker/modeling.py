import os
from dataclasses import dataclass
from typing import List, Optional, TypeVar, Union

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BertPreTrainedModel,
    PreTrainedModel,
    RobertaModel,
)
from transformers.modeling_outputs import TokenClassifierOutput

from span_marker.configuration import SpanMarkerConfig
from span_marker.data.data_collator import SpanMarkerDataCollator
from span_marker.tokenizer import SpanMarkerTokenizer

T = TypeVar("T")


@dataclass
class SpanMarkerOutput(TokenClassifierOutput):
    num_words: Optional[torch.Tensor] = None


class SpanMarkerModel(PreTrainedModel):
    config_class = SpanMarkerConfig
    base_model_prefix = "encoder"

    def __init__(self, config: SpanMarkerConfig, encoder=None):
        super().__init__(config)
        self.config = config
        # `encoder` will be specified if this Model is initializer via .from_pretrained with an encoder
        # If .from_pretrained is called with a SpanMarkerModel instance, then we use the "traditional"
        # PreTrainedModel.from_pretrained, which won't include an encoder keyword argument. In that case,
        # we must create an "empty" encoder for PreTrainedModel.from_pretrained to fill with the correct
        # weights.
        if encoder is None:
            # Load the encoder via the Config to prevent having to use AutoModel.from_pretrained, which
            # could load e.g. all of `roberta-large` from the Hub unnecessarily.
            # However, use the SpanMarkerModel updated vocab_size
            encoder_config = AutoConfig.from_pretrained(
                self.config.encoder["_name_or_path"],
                **self.config.encoder,
            )
            encoder = AutoModel.from_config(encoder_config)
        self.encoder = encoder

        if self.config.hidden_dropout_prob:
            self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        else:
            self.dropout = nn.Identity()
        # TODO: hidden_size is not always defined
        self.classifier = nn.Linear((self.config.hidden_size or 768) * 2, self.config.num_labels)

        self.loss_func = nn.CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        num_words: Optional[torch.Tensor] = None,
    ) -> SpanMarkerOutput:
        token_type_ids = torch.zeros_like(input_ids)
        # TODO: Change position_ids from int64 to int8?
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        last_hidden_state = outputs[0]
        last_hidden_state = self.dropout(last_hidden_state)

        sequence_length = last_hidden_state.size(1)
        start_marker_idx = sequence_length - 2 * self.config.max_marker_length
        end_marker_idx = start_marker_idx + self.config.max_marker_length
        # The start marker embeddings concatenated with the end marker embeddings
        # TODO: Can we use view, which may be more efficient?
        # Answer: Yes, but only if we move to using start-end-...-start-end instead
        # of start-start-...-end-end.

        feature_vector = torch.cat(
            (
                last_hidden_state[:, start_marker_idx:end_marker_idx],
                last_hidden_state[:, end_marker_idx:],
            ),
            dim=-1,
        )
        # NOTE: This was wrong in the older tests
        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)

        if labels is not None:
            loss = self.loss_func(logits.view(-1, self.config.num_labels), labels.view(-1))

        return SpanMarkerOutput(
            loss=loss if labels is not None else None, logits=logits, *outputs[2:], num_words=num_words
        )

    @classmethod
    def from_pretrained(
        cls: T, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, labels=None, **kwargs
    ) -> T:
        config_kwargs = {}
        # TODO: Consider moving where labels must be defined to elsewhere
        # TODO: Ensure that the provided labels match the labels in the config
        if labels is not None:
            # if "id2label" not in kwargs:
            config_kwargs["id2label"] = dict(enumerate(labels))
            # if "label2id" not in kwargs:
            config_kwargs["label2id"] = {v: k for k, v in config_kwargs["id2label"].items()}
            config_kwargs["num_labels"] = len(labels)
        else:
            raise Exception("Must provide `labels` list to `SpanMarkerModel.from_pretrained`.")

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs, **config_kwargs)

        # if 'pretrained_model_name_or_path' refers to a SpanMarkerModel instance
        if isinstance(config, cls.config_class):
            return super().from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)

        # If 'pretrained_model_name_or_path' refers to an encoder (roberta, bert, distilbert, electra, etc.)
        encoder = AutoModel.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        span_marker_config = cls.config_class(encoder_config=config.to_dict())
        model = cls(span_marker_config, encoder, *model_args, **kwargs)
        return model

    def predict(self, inputs: Union[str, List[str], List[List[str]]], tokenizer: SpanMarkerTokenizer):
        # breakpoint()
        inputs = tokenizer(inputs, config=self.config)
        inputs = [{key: value[idx] for key, value in inputs.items()} for idx in range(len(inputs["input_ids"]))]
        inputs = SpanMarkerDataCollator(tokenizer, self.config.max_marker_length)(inputs)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        logits = self(**inputs)[0]
        labels = logits.argmax(-1)
        print(labels.argmax(-1))
        breakpoint()
        # if isinstance(inputs, str):
        #     return self._predict_one(inputs)
        # return [self._predict_one(string) for string in inputs]
