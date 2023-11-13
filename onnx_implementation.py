import sys
from pathlib import Path

from tqdm import trange

sys.path.append(str(Path(__file__).resolve().parent.parent))
import time
from optimum.utils import DummyTextInputGenerator
from optimum.utils.normalized_config import NormalizedTextConfig
from optimum.exporters.onnx.config import TextEncoderOnnxConfig
from typing import Dict, Union, Tuple, Any, List
from optimum.version import __version__
from optimum.exporters.onnx import export, validate_model_outputs
from span_marker import SpanMarkerModel, SpanMarkerConfig
from span_marker.data_collator import SpanMarkerDataCollator
from span_marker.tokenizer import SpanMarkerTokenizer
import onnxruntime as ort
import torch
import os
import numpy as np
from datasets import Dataset, disable_progress_bar, enable_progress_bar
import logging
from optimum.pipelines import pipeline


logger = logging.getLogger(__name__)


assert "1.13.2" == __version__
print(f"Optimum version: {__version__}")


class SpanMarkerDummyTextInputenerator(DummyTextInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "input_ids",
        "attention_mask",
        "position_ids",
        "start_marker_indices",
        "num_marker_pairs",
        "num_words",
        "document_ids",
        "sentence_ids",
    )

    def __init__(self, *args, **kwargs):
        super(SpanMarkerDummyTextInputenerator, self).__init__(*args, **kwargs)
        self.batch = 1
        self.sequence_length_encoder = 512

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        min_value = 0
        if input_name == "input_ids":
            max_value = self.vocab_size
        elif input_name == "attention_mask":
            max_value = 1
        else:
            max_value = 2
        if input_name in ["input_ids", "position_ids"]:
            shape = [self.batch, self.sequence_length_encoder]
        elif input_name == "attention_mask":
            shape = [self.batch, self.sequence_length_encoder, self.sequence_length_encoder]
        else:
            shape = [self.batch]

        return self.random_int_tensor(shape, max_value, min_value=min_value, framework=framework, dtype=int_dtype)


class SpanMarkerOnnxConfig(TextEncoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (SpanMarkerDummyTextInputenerator,)
    DEFAULT_ONNX_OPSET = 13
    ATOL_FOR_VALIDATION = 1e-4

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        dynamic_axis = {0: "batch_size"}
        return {
            "input_ids": dynamic_axis,
            "attention_mask": dynamic_axis,
            "position_ids": dynamic_axis,
            "start_marker_indices": dynamic_axis,
            "num_marker_pairs": dynamic_axis,
            "num_words": dynamic_axis,
            "document_ids": dynamic_axis,
            "sentence_ids": dynamic_axis,
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        dynamic_axis = {0: "batch_size"}
        return {
            "logits": dynamic_axis,
            "out_num_marker_pairs": dynamic_axis,
            "out_num_words": dynamic_axis,
            "out_document_ids": dynamic_axis,
            "out_sentence_ids": dynamic_axis,
        }


class SpanMarkerOnnxPipeline:
    INPUT_TYPES = Union[str, List[str], List[List[str]]]
    OUTPUT_TYPES = Union[List[Dict[str, Union[str, int, float]]], List[List[Dict[str, Union[str, int, float]]]]]

    def __init__(
        self,
        onnx_path: Union[str, os.PathLike],
        repo_id: Union[str, os.PathLike],
        ort_provider: str = "CPUExecutionProvider",
        batch_size: int = 4,
        show_progress_bar: bool = False,
        *args,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.config = SpanMarkerConfig.from_pretrained(repo_id)
        self.tokenizer = SpanMarkerTokenizer.from_pretrained(repo_id, config=self.config)
        self.data_collator = SpanMarkerDataCollator(
            tokenizer=self.tokenizer, marker_max_length=self.config.marker_max_length
        )
        self.ort_session = self.load_ort_session(onnx_path, ort_provider)

    def load_ort_session(self, onnx_path: Union[str, os.PathLike], ort_provider: str, **kwargs) -> ort.InferenceSession:
        ort_session = ort.InferenceSession(onnx_path, providers=[ort_provider])
        return ort_session

    def _sanitize_parameters(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        return {}, {}, {}

    def preprocess(self, inputs: INPUT_TYPES) -> INPUT_TYPES:
        return inputs

    def _forward(self, inputs: INPUT_TYPES) -> OUTPUT_TYPES:
        from span_marker.trainer import Trainer

        if not inputs:
            return []

        # Track whether the input was a string sentence or a list of tokens
        single_input = False
        # Check if inputs is a string, i.e. a string sentence, or
        # if it is a list of strings without spaces, i.e. if it's 1 tokenized sentence
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(element, str) and " " not in element for element in inputs)
        ):
            single_input = True
            dataset = Dataset.from_dict({"tokens": [inputs]})

        # Otherwise, we likely have a list of strings, i.e. a list of string sentences,
        # or a list of lists of strings, i.e. a list of tokenized sentences
        # if isinstance(inputs, list) and all(isinstance(element, str) and " " not in element for element in inputs):
        # return [self._predict_one(sentence) for sentence in inputs]
        elif isinstance(inputs, list):
            dataset = Dataset.from_dict({"tokens": inputs})

        elif isinstance(inputs, Dataset):
            dataset = inputs

        else:
            raise ValueError(
                "`SpanMarkerModel.predict` could not recognize your input. It accepts the following:\n"
                "* str: a string sentence.\n"
                "* List[str]: a pre-tokenized string sentence, i.e. a list of words.\n"
                "* List[str]: a list of multiple string sentences.\n"
                "* List[List[str]]: a list of multiple pre-tokenized string sentences, i.e. a list with lists of words.\n"
                "* Dataset: A 🤗 Dataset with `tokens` column and optionally `document_id` and `sentence_id` columns.\n"
                "    If the optional columns are provided, they will be used to provide document-level context."
            )

        if "document_id" not in dataset.column_names:
            tokens = [dataset["tokens"]]
            data_dict = {
                "tokens": [],
                "document_id": [],
                "sentence_id": [],
            }
            for document_id, document in enumerate(tokens):
                for sentence_id, sentence in enumerate(document):
                    data_dict["document_id"].append(document_id)
                    data_dict["sentence_id"].append(sentence_id)
                    data_dict["tokens"].append(sentence)
            dataset = Dataset.from_dict(data_dict)

        dataset = dataset.remove_columns(set(dataset.column_names) - {"tokens", "document_id", "sentence_id"})
        num_inputs = len(dataset)
        dataset: Dataset = dataset.add_column("id", range(num_inputs))
        results = [
            {
                "tokens": tokens,
                "scores": [],
                "labels": [],
                "num_words": None,
            }
            for tokens in dataset["tokens"]
        ]

        # Tokenize & add start/end markers
        tokenizer_dict = self.tokenizer(
            {"tokens": dataset["tokens"]}, return_num_words=True, return_batch_encoding=True
        )
        batch_encoding = tokenizer_dict.pop("batch_encoding")
        dataset = dataset.remove_columns("tokens")
        for key, value in tokenizer_dict.items():
            dataset = dataset.add_column(key, value)

        # Add context if possible
        if {"document_id", "sentence_id"} <= set(dataset.column_names):
            if not self.config.trained_with_document_context:
                logger.warning(
                    "This model was trained without document-level context: "
                    "inference with document-level context may cause decreased performance."
                )
            # Add column to be able to revert sorting later
            dataset = dataset.add_column("__sort_id", range(len(dataset)))
            # Sorting by doc ID and then sentence ID is required for add_context
            dataset = dataset.sort(column_names=["document_id", "sentence_id"])
            dataset = Trainer.add_context(
                dataset,
                self.tokenizer.model_max_length,
                max_prev_context=self.config.max_prev_context,
                max_next_context=self.config.max_next_context,
                show_progress_bar=self.show_progress_bar,
            )
            dataset = dataset.sort(column_names=["__sort_id"])
            dataset = dataset.remove_columns("__sort_id")
        elif self.config.trained_with_document_context:
            logger.warning(
                "This model was trained with document-level context: "
                "inference without document-level context may cause decreased performance."
            )

        if not self.show_progress_bar:
            disable_progress_bar()

        dataset = dataset.map(
            Trainer.spread_sample,
            batched=True,
            desc="Spreading data between multiple samples",
            fn_kwargs={
                "model_max_length": self.tokenizer.model_max_length,
                "marker_max_length": self.config.marker_max_length,
            },
        )
        if not self.show_progress_bar:
            enable_progress_bar()
        for batch_start_idx in trange(0, len(dataset), self.batch_size, leave=True, disable=not self.show_progress_bar):
            batch = dataset.select(range(batch_start_idx, min(len(dataset), batch_start_idx + self.batch_size)))
            # Expanding the small tokenized output into full-scale input_ids, position_ids and attention_mask matrices.
            batch = self.data_collator(batch)
            # Moving the inputs to the right device with onnx format
            onnx_input = {key: value.detach().numpy().astype(np.int64) for key, value in batch.items()}
            ort_output = self.ort_session.run(output_names=["logits"], input_feed=onnx_input)
            logits = ort_output[0]
            out_num_marker_pairs = batch["num_marker_pairs"]
            out_num_words = batch["num_words"]
            # Computing probabilities based on the logits
            probs = torch.tensor(logits).softmax(-1)
            # Get the labels and the correponding probability scores
            scores, labels = probs.max(-1)
            # TODO: Iterate over output.num_marker_pairs instead with enumerate
            for iter_idx in range(out_num_marker_pairs.size(0)):
                input_id = dataset["id"][batch_start_idx + iter_idx]
                num_marker_pairs = out_num_marker_pairs[iter_idx]
                results[input_id]["scores"].extend(scores[iter_idx, :num_marker_pairs].tolist())
                results[input_id]["labels"].extend(labels[iter_idx, :num_marker_pairs].tolist())
                results[input_id]["num_words"] = out_num_words[iter_idx]

        all_entities = []
        id2label = self.config.id2label
        for sample_idx, sample in enumerate(results):
            scores = sample["scores"]
            labels = sample["labels"]
            num_words = sample["num_words"]
            sentence = sample["tokens"]
            # Get all of the valid spans to match with the score and labels
            spans = list(self.tokenizer.get_all_valid_spans(num_words, self.config.entity_max_length))

            word_selected = [False] * num_words
            sentence_entities = []
            assert len(spans) == len(scores) and len(spans) == len(labels)
            for (word_start_index, word_end_index), score, label_id in sorted(
                zip(spans, scores, labels), key=lambda tup: tup[1], reverse=True
            ):
                if label_id != self.config.outside_id and not any(word_selected[word_start_index:word_end_index]):
                    char_start_index = batch_encoding.word_to_chars(sample_idx, word_start_index).start
                    char_end_index = batch_encoding.word_to_chars(sample_idx, word_end_index - 1).end
                    entity = {
                        "span": sentence[char_start_index:char_end_index]
                        if isinstance(sentence, str)
                        else sentence[word_start_index:word_end_index],
                        "label": id2label[label_id],
                        "score": score,
                    }
                    if isinstance(sentence, str):
                        entity["char_start_index"] = char_start_index
                        entity["char_end_index"] = char_end_index
                    else:
                        entity["word_start_index"] = word_start_index
                        entity["word_end_index"] = word_end_index
                    sentence_entities.append(entity)

                    word_selected[word_start_index:word_end_index] = [True] * (word_end_index - word_start_index)
            all_entities.append(
                sorted(
                    sentence_entities,
                    key=lambda entity: entity["char_start_index"]
                    if isinstance(sentence, str)
                    else entity["word_start_index"],
                )
            )
        # if the input was a string or a list of tokens, return a list of dictionaries
        if single_input and len(all_entities) == 1:
            return all_entities[0]
        return all_entities

    def postprocess(self, outputs: OUTPUT_TYPES) -> OUTPUT_TYPES:
        return outputs

    def __call__(self, raw_inputs: INPUT_TYPES) -> OUTPUT_TYPES:
        inputs = self.preprocess(raw_inputs)
        raw_outputs = self._forward(inputs)
        outputs = self.postprocess(raw_outputs)
        return outputs


if __name__ == "__main__":
    # Load SpanMarker model
    repo_id = "lxyuan/span-marker-bert-base-multilingual-uncased-multinerd"
    base_model = SpanMarkerModel.from_pretrained(repo_id)
    base_model_config = base_model.config

    #  Export to onnx
    onnx_path = Path("spanmarker_model.onnx")
    onnx_config = SpanMarkerOnnxConfig(base_model_config)
    onnx_inputs, onnx_outputs = export(
        base_model,
        SpanMarkerOnnxConfig(base_model.config, task="token-classification"),
        onnx_path,
        opset=13,
        model_kwargs={"dynamic_axes": True},
    )

    # ONNX Validation
    validate_model_outputs(onnx_config, base_model, onnx_path, onnx_outputs, onnx_config.ATOL_FOR_VALIDATION)

    # Load ONNX Pipeline
    onnx_path = Path("spanmarker_model.onnx")
    repo_id = "lxyuan/span-marker-bert-base-multilingual-uncased-multinerd"
    onnx_pipe = SpanMarkerOnnxPipeline(onnx_path=onnx_path, repo_id=repo_id)

    sample = ["Pedro is working in Alicante"]  # It works
    sample = [
        "Pedro is working in Alicante. Pedro is working in Alicante. Pedro is working in Alicante.Pedro is working in Alicante. Pedro is working in Alicante. Pedro is working in Alicante.Pedro is working in Alicante. Pedro is working in Alicante. Pedro is working in Alicante",
    ]  # It doesn't work
    start_time = time.time()
    onnx_pipe(sample)
    end_time = time.time()
    print(f"Execution time: {end_time-start_time}")
