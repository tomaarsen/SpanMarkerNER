import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    PreTrainedModel,
)
from transformers.modeling_outputs import TokenClassifierOutput

from span_marker.configuration import SpanMarkerConfig
from span_marker.data_collator import SpanMarkerDataCollator
from span_marker.tokenizer import SpanMarkerTokenizer


@dataclass
class SpanMarkerOutput(TokenClassifierOutput):
    num_words: Optional[torch.Tensor] = None


class SpanMarkerModel(PreTrainedModel):
    config_class = SpanMarkerConfig
    base_model_prefix = "encoder"

    def __init__(self, config: SpanMarkerConfig, encoder=None, **kwargs):
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
            encoder_config = AutoConfig.from_pretrained(self.config.encoder["_name_or_path"], **self.config.encoder)
            encoder = AutoModel.from_config(encoder_config)
        self.encoder = encoder

        dropout_rate = self.config.get(["hidden_dropout_prob", "dropout_rate"], default=0.1)
        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = nn.Identity()
        # TODO: Get a less arbitrary default
        hidden_size = self.config.get("hidden_size", default=768)
        self.classifier = nn.Linear(hidden_size * 2, self.config.num_labels)
        self.loss_func = nn.CrossEntropyLoss()

        # tokenizer and data collator are filled using set_tokenizer
        self.tokenizer = None
        self.data_collator = None

        # Initialize weights and apply final processing
        self.post_init()

    def set_tokenizer(self, tokenizer: SpanMarkerTokenizer) -> None:
        self.tokenizer = tokenizer
        self.data_collator = SpanMarkerDataCollator(
            tokenizer=tokenizer, marker_max_length=self.config.marker_max_length
        )

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            initializer_range = self.config.get("initializer_range", default=0.02)
            module.weight.data.normal_(mean=0.0, std=initializer_range)
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
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        last_hidden_state = outputs[0]
        last_hidden_state = self.dropout(last_hidden_state)

        sequence_length = last_hidden_state.size(1)
        start_marker_idx = sequence_length - 2 * self.config.marker_max_length
        end_marker_idx = start_marker_idx + self.config.marker_max_length
        # TODO: Can we use view, which may be more efficient?
        # Answer: Yes, but only if we move to using start-end-...-start-end instead
        # of start-start-...-end-end.

        # The start marker embeddings concatenated with the end marker embeddings
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
        cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, labels=None, **kwargs
    ) -> "SpanMarkerModel":
        config_kwargs = {}
        # TODO: Consider moving where labels must be defined to elsewhere
        # TODO: Ensure that the provided labels match the labels in the config
        if labels is not None:
            # if "id2label" not in kwargs:
            config_kwargs["id2label"] = dict(enumerate(labels))
            # if "label2id" not in kwargs:
            config_kwargs["label2id"] = {v: k for k, v in config_kwargs["id2label"].items()}

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs, **config_kwargs)

        # if 'pretrained_model_name_or_path' refers to a SpanMarkerModel instance
        if isinstance(config, cls.config_class):
            model = super().from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)

        # If 'pretrained_model_name_or_path' refers to an encoder (roberta, bert, distilbert, electra, etc.)
        else:
            encoder = AutoModel.from_pretrained(pretrained_model_name_or_path, *model_args, config=config)
            config = cls.config_class(encoder_config=config.to_dict())
            model = cls(config, encoder, *model_args, **kwargs)

        # Pass the tokenizer directly to the model for convenience
        tokenizer = SpanMarkerTokenizer.from_pretrained(pretrained_model_name_or_path, config=config, **kwargs)
        model.set_tokenizer(tokenizer)
        model.resize_token_embeddings(len(tokenizer))

        return model

    def predict_one(
        self, sentence: Union[str, List[str]], allow_overlapping: bool = False
    ) -> List[Dict[str, Union[str, int, float]]]:
        # Tokenization, i.e. computing spans, adding span markers to position_ids
        tokenized = self.tokenizer(sentence, return_num_words=True, return_batch_encoding=True)
        num_words = tokenized.pop("num_words")[0]
        batch_encoding = tokenized.pop("batch_encoding")
        # Converting into a common batch format like the data collator wants
        tokenized = [
            {key: value[idx] for key, value in tokenized.items()} for idx in range(len(tokenized["input_ids"]))
        ]
        # Expanding the small tokenized output into full-scale input_ids, position_ids and attention_mask matrices.
        collated = self.data_collator(tokenized)
        # Moving the inputs to the right device
        inputs = {key: value.to(self.device) for key, value in collated.items()}

        logits = self(**inputs)[0]
        # Computing probabilities based on the logits
        probs = logits.softmax(-1)
        # Get the labels and the correponding probability scores
        scores, labels = probs.max(-1)
        # Reduce the dimensionality and convert to normal Python lists
        scores = scores.view(-1).tolist()
        labels = labels.view(-1).tolist()
        # Get all of the valid spans to match with the score and labels
        spans = list(self.tokenizer.get_all_valid_spans(num_words, self.config.entity_max_length))

        output = []
        id2label = self.config.id2label
        if self.config.are_labels_schemed():
            id2label = {label_id: id2label[self.config.id2reduced_id[label_id]] for label_id in self.config.id2label}
        # If we don't allow overlapping, then we keep track of a boolean for each word, indicating if it has been
        # selected already by a previous, higher score entity span
        if not allow_overlapping:
            word_selected = [False] * num_words
        for (word_start_index, word_end_index), score, label_id in sorted(
            zip(spans, scores, labels), key=lambda tup: tup[1], reverse=True
        ):
            if label_id != self.config.outside_id and (
                allow_overlapping or not any(word_selected[word_start_index:word_end_index])
            ):
                char_start_index = batch_encoding.word_to_chars(0, word_start_index).start
                char_end_index = batch_encoding.word_to_chars(0, word_end_index - 1).end
                output.append(
                    {
                        "word_start_index": word_start_index,
                        "word_end_index": word_end_index,
                        "char_start_index": char_start_index,
                        "char_end_index": char_end_index,
                        "label": id2label[str(label_id)],
                        "score": score,
                        "span": sentence[char_start_index:char_end_index]
                        if isinstance(sentence, str)
                        else sentence[word_start_index:word_end_index],
                    }
                )
                if not allow_overlapping:
                    word_selected[word_start_index:word_end_index] = [True] * (word_end_index - word_start_index)
        return sorted(output, key=lambda entity: entity["word_start_index"])

    def predict(
        self, inputs: Union[str, List[str], List[List[str]]], allow_overlapping: bool = False
    ) -> Union[List[Dict[str, Union[str, int, float]]], List[List[Dict[str, Union[str, int, float]]]]]:
        if not inputs:
            return []

        # Check if inputs is a string, i.e. a string sentence, or
        # if it is a list of strings without spaces, i.e. if it's 1 tokenized sentence
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(element, str) and " " not in element for element in inputs)
        ):
            return self.predict_one(inputs, allow_overlapping=allow_overlapping)

        # Otherwise, we likely have a list of strings, i.e. a list of string sentences,
        # or a list of lists of strings, i.e. a list of tokenized sentences
        # if isinstance(inputs, list) and all(isinstance(element, str) and " " not in element for element in inputs):
        return [self.predict_one(sentence) for sentence in inputs]
