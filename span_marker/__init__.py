__version__ = "1.1.2.dev"

import logging

from transformers import AutoConfig, AutoModel, TrainingArguments

from span_marker.configuration import SpanMarkerConfig
from span_marker.modeling import SpanMarkerModel
from span_marker.trainer import Trainer

AutoConfig.register("span-marker", SpanMarkerConfig)
AutoModel.register(SpanMarkerConfig, SpanMarkerModel)

logger = logging.getLogger("span_marker")
logger.setLevel(logging.INFO)
