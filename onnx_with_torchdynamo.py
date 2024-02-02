import sys
from pathlib import Path
import multiprocessing
from tqdm import trange
import time
from typing import Any, Dict, Optional, Union, List

sys.path.append(str(Path(__file__).resolve().parent.parent))

from span_marker import SpanMarkerModel, SpanMarkerConfig
from span_marker.data_collator import SpanMarkerDataCollator
from optimum.utils import DummyTextInputGenerator
from transformers import AutoConfig, AutoModel

from span_marker.output import SpanMarkerOutput
from span_marker.tokenizer import SpanMarkerTokenizer
import onnxruntime as ort
import torch
import os
import numpy as np
import torch.nn.functional as F
from optimum.version import __version__ as optimum_version

import logging

logger = logging.getLogger(__name__)

print(f"Onnxruntime version: {ort.__version__}")

ORT_OPSET = 13


class SpanMarkerDummyInputenerator:
    SUPPORTED_INPUT_NAMES = ["input_ids", "attention_mask", "position_ids", "start_marker_indices", "num_marker_pairs"]
    BATCH_SIZE = 1

    @classmethod
    def generate_dummy_input(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], framework: str = "pt", dtype: str = torch.int32
    ):
        config = SpanMarkerConfig.from_pretrained(pretrained_model_name_or_path)
        vocab_size = config.vocab_size
        sequence_length = config.model_max_length_default

        dummy_input = {}
        values = {
            "input_ids": {"max": vocab_size, "min": 0, "shape": [cls.BATCH_SIZE, sequence_length]},
            "attention_mask": {
                "max": 1,
                "min": 0,
                "shape": [cls.BATCH_SIZE, sequence_length, sequence_length],
            },
            "position_ids": {
                "max": sequence_length,
                "min": 0,
                "shape": [cls.BATCH_SIZE, sequence_length],
            },
            "start_marker_indices": {"max": 10, "min": 0, "shape": [cls.BATCH_SIZE]},
            "num_marker_pairs": {"max": 10, "min": 0, "shape": [cls.BATCH_SIZE]},
        }

        for value in cls.SUPPORTED_INPUT_NAMES:
            min_val = values[value]["min"]
            max_value = values[value]["max"]
            shape = values[value]["shape"]
            dummy_input[value] = torch.randint(low=min_val, high=max_value, size=shape, dtype=dtype)
        return dummy_input


class SpanMarkerArchitecture(torch.nn.Module):
    def __init__(self, repo_id: str):
        super(SpanMarkerArchitecture, self).__init__()
        self.encoder = SpanMarkerModel.from_pretrained(repo_id).encoder
        self.classifier = SpanMarkerModel.from_pretrained(repo_id).classifier

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        start_marker_indices: torch.Tensor,
        num_marker_pairs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward call of the SpanMarkerModel.
        Args:
            input_ids (~torch.Tensor): Input IDs including start/end markers.
            attention_mask (~torch.Tensor): Attention mask matrix including one-directional attention for markers.
            position_ids (~torch.Tensor): Position IDs including start/end markers.
        None.

        Returns:
            outputs: Encoder outputs
        """
        token_type_ids = torch.zeros_like(input_ids)
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        last_hidden_state = outputs[0]
        sequence_length = last_hidden_state.size(1)
        batch_size = last_hidden_state.size(0)
        # Get the indices where the end markers start
        end_marker_indices = start_marker_indices + num_marker_pairs
        sequence_length_last_hidden_state = last_hidden_state.size(2) * 2
        #  Pre-allocates the necessary space for feature_vector
        feature_vector = torch.zeros(batch_size, sequence_length // 2, sequence_length_last_hidden_state)
        for i in range(batch_size):
            feature_vector[
                i, : end_marker_indices[i] - start_marker_indices[i], : last_hidden_state.shape[-1]
            ] = last_hidden_state[i, start_marker_indices[i] : end_marker_indices[i]]
            feature_vector[
                i, : end_marker_indices[i] - start_marker_indices[i], last_hidden_state.shape[-1] :
            ] = last_hidden_state[i, end_marker_indices[i] : end_marker_indices[i] + num_marker_pairs[i]]

        logits = self.classifier(feature_vector)
        return logits


repo_id = "lxyuan/span-marker-bert-base-multilingual-uncased-multinerd"
onnx_path = "spanmarker.onnx"
model = SpanMarkerArchitecture(repo_id)
dummy_input = SpanMarkerDummyInputenerator.generate_dummy_input(repo_id)
export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
onnx_program = torch.onnx.dynamo_export(model, export_options=export_options, **dummy_input)
