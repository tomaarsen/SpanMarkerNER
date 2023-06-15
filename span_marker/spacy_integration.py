import os
from typing import Any, Optional, Union

import torch
from datasets import Dataset
from spacy.tokens import Doc, Span

from span_marker.modeling import SpanMarkerModel


class SpacySpanMarkerWrapper:
    """A wrapper of SpanMarker for spaCy, allowing SpanMarker to be used as a spaCy pipeline component.

    The `span_marker` pipeline component sets the following extensions:

        * `Doc._.ents`: A tuple of `Span` instances, e.g. `(Cleopatra VII, Cleopatra the Great, 69 BCE)`
        * `Span._.label`: A string representing the entity label, e.g. `PERSON`
        * `Span._.score`: The SpanMarker confidence score for the prediction.

    Example::

        >>> import spacy
        >>> import span_marker
        >>> nlp = spacy.load("en_core_web_sm")
        >>> nlp.add_pipe("span_marker", config={"model": "tomaarsen/span-marker-roberta-large-ontonotes5"})
        >>> text = '''Cleopatra VII, also known as Cleopatra the Great, was the last active ruler of the
        ... Ptolemaic Kingdom of Egypt. She was born in 69 BCE and ruled Egypt from 51 BCE until her
        ... death in 30 BCE.'''
        >>> doc = nlp(text)
        >>> doc._.ents
        (Cleopatra VII, Cleopatra the Great, 69 BCE, Egypt, 51 BCE, 30 BCE)
        >>> for span in doc._.ents:
        ...     print((span, span._.label, span._.score))
        (Cleopatra VII, 'PERSON', 0.994394063949585)
        (Cleopatra the Great, 'PERSON', 0.9954156875610352)
        (69 BCE, 'DATE', 0.9956725835800171)
        (Egypt, 'GPE', 0.9962241649627686)
        (51 BCE, 'DATE', 0.9894670844078064)
        (30 BCE, 'DATE', 0.9939478635787964)
    """

    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *args,
        batch_size: int = 4,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> None:
        self.model = SpanMarkerModel.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if device:
            self.model.to(device)
        elif torch.cuda.is_available():
            self.model.to("cuda")
        self.batch_size = batch_size
        Doc.set_extension("ents", default=[])
        Span.set_extension("label", default="")
        Span.set_extension("score", default=0.0)

    def __call__(self, doc: Doc) -> Doc:
        """Fill the `doc._.ents`, `span._.label` and `span._.score` extensions using the SpanMarker model."""
        sents = list(doc.sents)
        inputs = [[token.text for token in sent] for sent in sents]
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
                span._.label = entity["label"]
                span._.score = entity["score"]
                outputs.append(span)

        doc._.ents = tuple(outputs)
        return doc
