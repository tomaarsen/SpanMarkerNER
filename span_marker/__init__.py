__version__ = "0.1.0"

from transformers import AutoConfig, AutoModel

from span_marker.configuration import SpanMarkerConfig
from span_marker.modeling import SpanMarkerModel
from span_marker.trainer import Trainer

AutoConfig.register("span-marker", SpanMarkerConfig)
AutoModel.register(SpanMarkerConfig, SpanMarkerModel)
