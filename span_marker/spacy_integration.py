import os
from typing import Any, Optional, Union

import torch
from datasets import Dataset
from spacy.tokens import Doc, Span

from span_marker.modeling import SpanMarkerModel


class SpacySpanMarkerWrapper:
    """This wrapper allows SpanMarker to be used as a drop-in replacement of the "ner" pipeline component.

    Usage:

    .. code-block:: diff

         import spacy

         nlp = spacy.load("en_core_web_sm")
       + nlp.add_pipe("span_marker", config={"model": "tomaarsen/span-marker-roberta-large-ontonotes5"})

         text = '''Cleopatra VII, also known as Cleopatra the Great, was the last active ruler of the
         Ptolemaic Kingdom of Egypt. She was born in 69 BCE and ruled Egypt from 51 BCE until her
         death in 30 BCE.'''
         doc = nlp(text)

    Example::

        >>> import spacy
        >>> import span_marker
        >>> nlp = spacy.load("en_core_web_sm", exclude=["ner"])
        >>> nlp.add_pipe("span_marker", config={"model": "tomaarsen/span-marker-roberta-large-ontonotes5"})
        >>> text = '''Cleopatra VII, also known as Cleopatra the Great, was the last active ruler of the
        ... Ptolemaic Kingdom of Egypt. She was born in 69 BCE and ruled Egypt from 51 BCE until her
        ... death in 30 BCE.'''
        >>> doc = nlp(text)
        >>> doc.ents
        (Cleopatra VII, Cleopatra the Great, 69 BCE, Egypt, 51 BCE, 30 BCE)
        >>> for span in doc.ents:
        ...     print((span, span.label_))
        (Cleopatra VII, 'PERSON')
        (Cleopatra the Great, 'PERSON')
        (69 BCE, 'DATE')
        (Egypt, 'GPE')
        (51 BCE, 'DATE')
        (30 BCE, 'DATE')
    """

    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *args,
        batch_size: int = 4,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> None:
        """Initialize a SpanMarker wrapper for spaCy.

        Args:
            pretrained_model_name_or_path (Union[str, os.PathLike]): The path to a locally pretrained SpanMarker model
                or a model name from the Hugging Face hub, e.g. `tomaarsen/span-marker-roberta-large-ontonotes5`
            batch_size (int): The number of samples to include per batch. Higher is faster, but requires more memory.
                Defaults to 4.
            device (Optional[Union[str, torch.device]]): The device to place the model on. Defaults to None.
        """
        self.model = SpanMarkerModel.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if device:
            self.model.to(device)
        elif torch.cuda.is_available():
            self.model.to("cuda")
        self.batch_size = batch_size

    def __call__(self, doc: Doc) -> Doc:
        """Fill `doc.ents` and `span.label_` using the chosen SpanMarker model."""
        sents = list(doc.sents)
        inputs = [[token.text if not token.is_space else "" for token in sent] for sent in sents]
        # use document-level context in the inference if the model was also trained that way
        if self.model.config.trained_with_document_context:
            inputs = Dataset.from_dict(
                {
                    "tokens": inputs,
                    "document_id": [0] * len(inputs),
                    "sentence_id": range(len(inputs)),
                }
            )
        outputs = []

        entities_list = self.model.predict(inputs, batch_size=self.batch_size)
        for sentence, entities in zip(sents, entities_list):
            for entity in entities:
                start = entity["word_start_index"]
                end = entity["word_end_index"]
                span = sentence[start:end]
                span.label_ = entity["label"]
                outputs.append(span)

        doc.set_ents(outputs)
        return doc
