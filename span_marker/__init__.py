__version__ = "1.4.0"

import importlib
import logging
import os
from typing import Optional, Union

import torch
from transformers import AutoConfig, AutoModel, TrainingArguments
from transformers.pipelines import PIPELINE_REGISTRY, pipeline

from span_marker.configuration import SpanMarkerConfig
from span_marker.model_card import SpanMarkerModelCardData
from span_marker.modeling import SpanMarkerModel
from span_marker.pipeline_component import SpanMarkerPipeline
from span_marker.trainer import Trainer

# Set up for Transformers
AutoConfig.register("span-marker", SpanMarkerConfig)
AutoModel.register(SpanMarkerConfig, SpanMarkerModel)
PIPELINE_REGISTRY.register_pipeline(
    "span-marker",
    pipeline_class=SpanMarkerPipeline,
    pt_model=SpanMarkerModel,
    type="text",
    default={"pt": ("tomaarsen/span-marker-bert-base-fewnerd-fine-super", "main")},
)

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
        "overwrite_entities": False,
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
        overwrite_entities: bool,
    ) -> SpacySpanMarkerWrapper:
        if overwrite_entities:
            # Remove the existing NER component, if it exists,
            # to allow for SpanMarker to act as a drop-in replacement
            try:
                nlp.remove_pipe("ner")
            except ValueError:
                # The `ner` pipeline component was not found
                pass
        return SpacySpanMarkerWrapper(model, batch_size=batch_size, device=device)


# If codecarbon is installed and the log level is not defined,
# automatically overwrite the default to "error"
if importlib.util.find_spec("codecarbon") and "CODECARBON_LOG_LEVEL" not in os.environ:
    os.environ["CODECARBON_LOG_LEVEL"] = "error"

logger = logging.getLogger("span_marker")
logger.setLevel(logging.INFO)
