__version__ = "1.2.4"

import logging
from typing import Optional, Union

import torch
from transformers import AutoConfig, AutoModel, TrainingArguments

from span_marker.configuration import SpanMarkerConfig
from span_marker.modeling import SpanMarkerModel
from span_marker.trainer import Trainer

# Set up for Transformers
AutoConfig.register("span-marker", SpanMarkerConfig)
AutoModel.register(SpanMarkerConfig, SpanMarkerModel)

# Set up for spaCy
try:
    from spacy.language import Language
except ImportError:
    pass
else:
    from span_marker.spacy_integration import SpacySpanMarkerWrapper

    DEFAULT_SPACY_CONFIG = {
        "model": "tomaarsen/span-marker-roberta-large-ontonotes5",
        "batch_size": 4,
        "device": None,
    }

    @Language.factory(
        "span_marker",
        assigns=["doc.ents", "token.ent_iob", "token.ent_type"],
        default_config=DEFAULT_SPACY_CONFIG,
    )
    def _spacy_span_marker_factory(
        nlp: Language,  # pylint: disable=W0613
        name: str,  # pylint: disable=W0613
        model: str,
        batch_size: int,
        device: Optional[Union[str, torch.device]],
    ) -> SpacySpanMarkerWrapper:
        # Remove the existing NER component, if it exists,
        # to allow for SpanMarker to act as a drop-in replacement
        try:
            nlp.remove_pipe("ner")
        except ValueError:
            # The `ner` pipeline component was not found
            pass
        return SpacySpanMarkerWrapper(model, batch_size=batch_size, device=device)


logger = logging.getLogger("span_marker")
logger.setLevel(logging.INFO)
