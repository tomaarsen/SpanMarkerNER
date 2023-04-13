import os
from typing import Callable, Dict, List, Optional, Type, TypeVar, Union

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

from span_marker import __version__ as span_marker_version
from span_marker.configuration import SpanMarkerConfig
from span_marker.data_collator import SpanMarkerDataCollator
from span_marker.model_card import MODEL_CARD_TEMPLATE
from span_marker.output import SpanMarkerOutput
from span_marker.tokenizer import SpanMarkerTokenizer

T = TypeVar("T", bound="SpanMarkerModel")


class SpanMarkerModel(PreTrainedModel):
    """
    This SpanMarker model allows for Named Entity Recognition (NER) using a variety of underlying encoders,
    such as BERT and RoBERTa. The model should be initialized using :meth:`~SpanMarkerModel.from_pretrained`,
    e.g. like so:

    >>> # Initialize a SpanMarkerModel using a pretrained encoder
    >>> model = SpanMarkerModel.from_pretrained("bert-base-cased", labels=["O", "B-PER", "I-PER", "B-ORG", "I-ORG", ...])
    >>> # Load a pretrained SpanMarker model
    >>> model = SpanMarkerModel.from_pretrained("tomaarsen/span-marker-bert-base-fewnerd-fine-super")

    After the model is loaded (and finetuned if it wasn't already), it can be used to predict entities:

    >>> model.predict("A prototype was fitted in the mid-'60s in a one-off DB5 extended 4'' after the doors and "
    ... "driven by Marek personally, and a normally 6-cylinder Aston Martin DB7 was equipped with a V8 unit in 1998.")
    [{'span': 'DB5', 'label': 'product-car', 'score': 0.8675689101219177, 'char_start_index': 52, 'char_end_index': 55},
     {'span': 'Marek', 'label': 'person-other', 'score': 0.9100819230079651, 'char_start_index': 99, 'char_end_index': 104},
     {'span': 'Aston Martin DB7', 'label': 'product-car', 'score': 0.9931442737579346, 'char_start_index': 143, 'char_end_index': 159}]
    """

    config_class = SpanMarkerConfig
    base_model_prefix = "encoder"

    def __init__(self, config: SpanMarkerConfig, encoder: Optional[PreTrainedModel] = None, **kwargs) -> None:
        """Initialize a SpanMarkerModel using configuration.

        Do not manually initialize a SpanMarkerModel this way! Use :meth:`~SpanMarkerModel.from_pretrained` instead.

        Args:
            config (SpanMarkerConfig): The configuration for this model.
            encoder (Optional[PreTrainedModel]): A PreTrainedModel acting as the underlying encoder.
                Defaults to None.
        """
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

    def _init_weights(self, module) -> None:
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
        """Forward call of the SpanMarkerModel.

        Args:
            input_ids (~torch.Tensor): Input IDs including start/end markers.
            attention_mask (~torch.Tensor): Attention mask matrix including one-directional attention for markers.
            position_ids (~torch.Tensor): Position IDs including start/end markers.
            labels (Optional[~torch.Tensor]): The labels for each span candidate. Defaults to None.
            num_words (Optional[~torch.Tensor]): The number of words for each batch sample. Defaults to None.

        Returns:
            SpanMarkerOutput: The output dataclass.
        """
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
        cls: Type[T],
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *model_args,
        labels: Optional[List[str]] = None,
        **kwargs,
    ) -> T:
        """Instantiate a pretrained pytorch model from a pre-trained model configuration.

        Example:

            >>> # Initialize a SpanMarkerModel using a pretrained encoder
            >>> model = SpanMarkerModel.from_pretrained("bert-base-cased", labels=["O", "B-PER", "I-PER", "B-ORG", "I-ORG", ...])
            >>> # Load a pretrained SpanMarker model
            >>> model = SpanMarkerModel.from_pretrained("tomaarsen/span-marker-bert-base-fewnerd-fine-super")

        Args:
            pretrained_model_name_or_path (Union[str, os.PathLike]):
                Either a pretrained encoder (e.g. ``bert-base-cased``, ``roberta-large``, etc.), or a pretrained SpanMarkerModel.
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under a
                      user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a *directory* containing model weights saved using
                      :meth:`SpanMarkerModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, ``./tf_model/model.ckpt.index``). In
                      this case, ``from_tf`` should be set to ``True`` and a configuration object should be provided as
                      ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                    - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
                      ``./flax_model/`` containing ``flax_model.msgpack``). In this case, ``from_flax`` should be set to
                      ``True``.

            labels (List[str], optional): A list of string labels corresponding to the ``ner_tags`` in your datasets.
                Only necessary when loading a SpanMarker model using a pretrained encoder. Defaults to None.

        Additional arguments are passed to :class:`~span_marker.configuration.SpanMarkerConfig` and the ``from_pretrained`` methods of
        :class:`~transformers.AutoConfig`, :class:`~transformers.AutoModel` and :class:`~span_marker.tokenizer.SpanMarkerTokenizer`.

        Returns:
            SpanMarkerModel: A :class:`SpanMarkerModel` instance, either ready for training using the :class:`Trainer` or\
                for inference via :meth:`SpanMarkerModel.predict`.
        """
        # If loading a SpanMarkerConfig, then we don't want to override id2label and label2id
        # Create an encoder or SpanMarker config
        config: PretrainedConfig = AutoConfig.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # if 'pretrained_model_name_or_path' refers to a SpanMarkerModel instance, initialize it directly
        if isinstance(config, cls.config_class):
            model = super().from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)

        # If 'pretrained_model_name_or_path' refers to an encoder (roberta, bert, distilbert, electra, etc.),
        # then initialize it and create the SpanMarker config and model using the encoder and its config.
        else:
            encoder = AutoModel.from_pretrained(pretrained_model_name_or_path, *model_args, config=config)
            if labels is None:
                raise ValueError(
                    "Please provide a `labels` list to `SpanMarkerModel.from_pretrained()`, e.g.\n"
                    ">>> SpanMarkerModel.from_pretrained(\n"
                    f'...     "{pretrained_model_name_or_path}",\n'
                    '...     labels=["O", "B-PER", "I-PER", "B-ORG", "I-ORG", ...]\n'
                    "... )\n"
                    "or\n"
                    ">>> SpanMarkerModel.from_pretrained(\n"
                    f'...     "{pretrained_model_name_or_path}",\n'
                    '...     labels=["O", "PER", "ORG", "LOC", "MISC"]\n'
                    "... )"
                )
            config.id2label = dict(enumerate(labels))
            config.label2id = {v: k for k, v in config.id2label.items()}
            # Set the span_marker version for freshly initialized models
            config = cls.config_class(
                encoder_config=config.to_dict(), span_marker_version=span_marker_version, **kwargs
            )
            model = cls(config, encoder, *model_args, **kwargs)

        # Pass the tokenizer directly to the model for convenience, this way the user doesn't have to
        # make it themselves.
        tokenizer = SpanMarkerTokenizer.from_pretrained(
            config.encoder.get("_name_or_path", pretrained_model_name_or_path), config=config, **kwargs
        )
        model.set_tokenizer(tokenizer)
        model.resize_token_embeddings(len(tokenizer))

        return model

    def predict(
        self, inputs: Union[str, List[str], List[List[str]]], allow_overlapping: bool = False
    ) -> Union[List[Dict[str, Union[str, int, float]]], List[List[Dict[str, Union[str, int, float]]]]]:
        """Predict named entities from input texts.

        Example::

            >>> model = SpanMarkerModel.from_pretrained(...)
            >>> model.predict("Amelia Earhart flew her single engine Lockheed Vega 5B across the Atlantic to Paris.")
            [{'span': 'Amelia Earhart', 'label': 'person-other', 'score': 0.7629689574241638, 'char_start_index': 0, 'char_end_index': 14},
             {'span': 'Lockheed Vega 5B', 'label': 'product-airplane', 'score': 0.9833564758300781, 'char_start_index': 38, 'char_end_index': 54},
             {'span': 'Atlantic', 'label': 'location-bodiesofwater', 'score': 0.7621214389801025, 'char_start_index': 66, 'char_end_index': 74},
             {'span': 'Paris', 'label': 'location-GPE', 'score': 0.9807717204093933, 'char_start_index': 78, 'char_end_index': 83}]
            >>> model.predict(['Caesar', 'led', 'the', 'Roman', 'armies', 'in', 'the', 'Gallic', 'Wars', 'before', 'defeating', 'his', 'political', 'rival', 'Pompey', 'in', 'a', 'civil', 'war'])
            [{'span': ['Caesar'], 'label': 'person-politician', 'score': 0.683479905128479, 'word_start_index': 0, 'word_end_index': 1},
             {'span': ['Roman'], 'label': 'location-GPE', 'score': 0.7114525437355042, 'word_start_index': 3, 'word_end_index': 4},
             {'span': ['Gallic', 'Wars'], 'label': 'event-attack/battle/war/militaryconflict', 'score': 0.9015670418739319, 'word_start_index': 7, 'word_end_index': 9},
             {'span': ['Pompey'], 'label': 'person-politician', 'score': 0.9601260423660278, 'word_start_index': 14, 'word_end_index': 15}]

        Args:
            inputs (Union[str, List[str], List[List[str]]]): Input sentences from which to extract entities.
                Valid datastructures are:

                * str: a string sentence.
                * List[str]: a pre-tokenized string sentence, i.e. a list of words.
                * List[str]: a list of multiple string sentences.
                * List[List[str]]: a list of multiple pre-tokenized string sentences, i.e. a list with lists of words.
            allow_overlapping (bool, optional): Whether to allow entity spans to overlap. The model does not
                have good support for this, so False is recommended. Defaults to False.

        Returns:
            Union[List[Dict[str, Union[str, int, float]]], List[List[Dict[str, Union[str, int, float]]]]]:
                If the input is a single sentence, then we output a list of dictionaries. Each dictionary
                represents one predicted entity, and contains the following keys:

                * ``label``: The predicted entity label.
                * ``span``: The text that the model deems an entity.
                * ``score``: The model its confidence.
                * ``word_start_index`` & ``word_end_index``: The word indices for the start/end of the entity,
                  if the input is pre-tokenized.
                * ``char_start_index`` & ``char_end_index``: The character indices for the start/end of the entity,
                  if the input is a string.

                If the input is multiple sentences, then we return a list containing multiple of the aforementioned lists.
        """
        if not inputs:
            return []

        # Check if inputs is a string, i.e. a string sentence, or
        # if it is a list of strings without spaces, i.e. if it's 1 tokenized sentence
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(element, str) and " " not in element for element in inputs)
        ):
            return self._predict_one(inputs, allow_overlapping=allow_overlapping)

        # Otherwise, we likely have a list of strings, i.e. a list of string sentences,
        # or a list of lists of strings, i.e. a list of tokenized sentences
        # if isinstance(inputs, list) and all(isinstance(element, str) and " " not in element for element in inputs):
        return [self._predict_one(sentence) for sentence in inputs]

    def _predict_one(
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
                        "span": sentence[char_start_index:char_end_index]
                        if isinstance(sentence, str)
                        else sentence[word_start_index:word_end_index],
                        "label": id2label[label_id],
                        "score": score,
                    }
                )
                if isinstance(sentence, str):
                    output[-1]["char_start_index"] = char_start_index
                    output[-1]["char_end_index"] = char_end_index
                else:
                    output[-1]["word_start_index"] = word_start_index
                    output[-1]["word_end_index"] = word_end_index

                if not allow_overlapping:
                    word_selected[word_start_index:word_end_index] = [True] * (word_end_index - word_start_index)
        return sorted(
            output,
            key=lambda entity: entity["char_start_index"] if isinstance(sentence, str) else entity["word_start_index"],
        )

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "10GB",
        safe_serialization: bool = False,
        variant: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().save_pretrained(
            save_directory,
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            variant=variant,
            **kwargs,
        )
        self.tokenizer.save_pretrained(
            save_directory,
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            variant=variant,
            **kwargs,
        )
        if "_name_or_path" in self.config.encoder:
            encoder_name_or_path = repr(self.config.encoder["_name_or_path"])
        else:
            encoder_name_or_path = "an unknown model"
        model_card_content = MODEL_CARD_TEMPLATE.format(
            model_name=save_directory, encoder_name_or_path=encoder_name_or_path
        )
        with open(os.path.join(save_directory, "README.md"), "w", encoding="utf-8") as f:
            f.write(model_card_content)
