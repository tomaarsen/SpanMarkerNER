__version__ = "1.1.2.dev"

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
        "model": "tomaarsen/span-marker-bert-tiny-fewnerd-coarse-super",
        "batch_size": 4,
        "device": None,
    }

    @Language.factory("span_marker", default_config=DEFAULT_SPACY_CONFIG)
    def _spacy_span_marker_factory(
        nlp: Language,  # pylint: disable=W0613
        name: str,  # pylint: disable=W0613
        model: str,
        batch_size: int,
        device: Optional[Union[str, torch.device]],
    ) -> SpacySpanMarkerWrapper:
        return SpacySpanMarkerWrapper(model, batch_size=batch_size, device=device)


logger = logging.getLogger("span_marker")
logger.setLevel(logging.INFO)
