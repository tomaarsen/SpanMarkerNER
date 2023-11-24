import sys
from pathlib import Path
import multiprocessing
from tqdm import trange
import time
from typing import Dict, Union, List

sys.path.append(str(Path(__file__).resolve().parent.parent))

from span_marker import SpanMarkerModel, SpanMarkerConfig
from span_marker.data_collator import SpanMarkerDataCollator
from optimum.utils import DummyTextInputGenerator
from transformers import AutoConfig,AutoModel

from span_marker.output import SpanMarkerOutput
from span_marker.tokenizer import SpanMarkerTokenizer
import onnxruntime as ort
import torch
import os
import numpy as np
import torch.nn.functional as F
from optimum.version import __version__ as optimum_version

from datasets import Dataset, disable_progress_bar, enable_progress_bar
import logging

logger = logging.getLogger(__name__)

print(f"Onnxruntime version: {ort.__version__}")

ORT_OPSET=13

class StackFeatureDummyInputGenerator():
    SUPPORTED_INPUT_NAMES = [
        "last_hidden_state",
        "start_marker_indices",
        "num_marker_pairs",
    ]
    BATCH_SIZE = 1

    @classmethod
    def generate_dummy_input(cls,pretrained_model_name_or_path:Union[str, os.PathLike],framework: str = "pt", dtype: str = torch.int32):
        
        config = SpanMarkerConfig.from_pretrained(pretrained_model_name_or_path)
        sequence_length = config.model_max_length_default
        hidden_size = config.hidden_size
        dummy_input = {}
        values = {
            "last_hidden_state": {"max": -2, "min": 1, "shape": [cls.BATCH_SIZE, sequence_length,hidden_size]},
            "start_marker_indices": {"max": sequence_length, "min": 0, "shape": [cls.BATCH_SIZE]},
            "num_marker_pairs": {"max": 10, "min": 0, "shape": [cls.BATCH_SIZE]},
        }

        for value in cls.SUPPORTED_INPUT_NAMES:
            min_val = values[value]["min"] if value in values else values["others"]["min"]
            max_value = values[value]["max"] if value in values else values["others"]["max"]
            shape = values[value]["shape"] if value in values else values["others"]["shape"]
            if value=="last_hidden_state":
                dtype = torch.float32
                dummy_input[value] = torch.randn(size=shape,dtype=dtype)
            else:
                dtype = torch.int32
                dummy_input[value] = torch.randint(low=min_val,high=max_value,size=shape,dtype=dtype)

            
             
        return dummy_input


class SpanMarkerDummyInputenerator():
    SUPPORTED_INPUT_NAMES = [
        "input_ids",
        "attention_mask",
        "position_ids",
        "start_marker_indices",
        "num_marker_pairs",
    ]
    BATCH_SIZE = 1

    @classmethod
    def generate_dummy_input(cls, pretrained_model_name_or_path:Union[str, os.PathLike],framework: str = "pt", dtype: str = torch.int32):
        
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
            "start_marker_indices": {"max": sequence_length, "min": 0, "shape": [cls.BATCH_SIZE]},
            "num_marker_pairs": {"max": 10, "min": 0, "shape": [cls.BATCH_SIZE]},
            "others": {"max": 1, "min": 0, "shape": [cls.BATCH_SIZE]},
        }

        for value in cls.SUPPORTED_INPUT_NAMES:
             min_val = values[value]["min"] if value in values else values["others"]["min"]
             max_value = values[value]["max"] if value in values else values["others"]["max"]
             shape = values[value]["shape"] if value in values else values["others"]["shape"]
             dummy_input[value] = torch.randint(low=min_val,high=max_value,size=shape,dtype=dtype)
             
        return dummy_input


class StackFeatureVector(torch.nn.Module):
    def forward(
        self,
        last_hidden_state:torch.Tensor,
        start_marker_indices:torch.Tensor,
        num_marker_pairs:torch.Tensor,
    ):
        sequence_length = last_hidden_state.size(1)
        batch_size = last_hidden_state.size(0)
        # Get the indices where the end markers start
        end_marker_indices = start_marker_indices + num_marker_pairs
        sequence_length_last_hidden_state = last_hidden_state.size(2) * 2
        
        feature_vector = torch.zeros(
            batch_size,
            sequence_length // 2,
            sequence_length_last_hidden_state,
        )

        for i in range(batch_size):
            feature_vector[i, :end_marker_indices[i]-start_marker_indices[i], :last_hidden_state.shape[-1]] = last_hidden_state[i, start_marker_indices[i] : end_marker_indices[i]]
            feature_vector[i, :end_marker_indices[i]-start_marker_indices[i], last_hidden_state.shape[-1]:] = last_hidden_state[i, end_marker_indices[i] : end_marker_indices[i] + num_marker_pairs[i]]
        return feature_vector
     

class SpanMarkerScriptModel(torch.nn.Module):
    def __init__(self, encoder,classifier,*args, **kwargs) -> None:
        super(SpanMarkerScriptModel,self).__init__(*args, **kwargs)
        self.encoder = encoder
        self.stack_feature_vector = torch.jit.script(StackFeatureVector(),tuple(dummy_input_stackfeature.values()))
        self.classifier = classifier
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
            
            #  Pre-allocates the necessary space for feature_vector
            feature_vector = self.stack_feature_vector(
                last_hidden_state,
                start_marker_indices,
                num_marker_pairs
            )
     
            logits = self.classifier(feature_vector)

            return logits




if __name__ == "__main__":
    # Load SpanMarker model
    repo_id = "lxyuan/span-marker-bert-base-multilingual-uncased-multinerd"
    base_model = SpanMarkerModel.from_pretrained(repo_id,torchscript=True)
    base_model.eval()
    base_model_config = base_model.config

    # Generate dummy inputs
    dummy_input = SpanMarkerDummyInputenerator.generate_dummy_input(repo_id)
    dummy_input_stackfeature = StackFeatureDummyInputGenerator.generate_dummy_input(repo_id)
    
    # Export SpanMaker model to TorchScript 
    spanmarker_script_model  = SpanMarkerScriptModel(encoder=base_model.encoder,classifier=base_model.classifier)
    spanmarker_script_model.eval()
    exported_traced_model = torch.jit.script(spanmarker_script_model,tuple(dummy_input.values()))

    # Export to onnx
    torch.onnx.export(exported_traced_model,
                    dummy_input,
                    "spanmarker.onnx",
                    input_names=[
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "start_marker_indices",
                    "num_marker_pairs"
                    ],
                    output_names=["logits"],
                    dynamic_axes={
                        "input_ids":{0:"batch_size"},
                        "attention_mask":{0:"batch_size"},
                        "position_ids":{0:"batch_size"},
                        "start_marker_indices":{0:"batch_size"},
                        "num_marker_pairs":{0:"batch_size"},
                        "logits":{0:"batch_size"}},
                    do_constant_folding=True,
                    export_params=True,
                    opset_version=ORT_OPSET,
                    )

