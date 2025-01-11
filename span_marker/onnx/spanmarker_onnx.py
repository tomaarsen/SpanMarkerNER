import multiprocessing
from tqdm import trange
from typing import Any, Dict, Optional, Union, List
from span_marker import SpanMarkerModel, SpanMarkerConfig
from span_marker.data_collator import SpanMarkerDataCollator
from span_marker.output import SpanMarkerOutput
from span_marker.tokenizer import SpanMarkerTokenizer
import onnxruntime as ort
from onnxruntime import SessionOptions
import pathlib
import onnx
from onnxconverter_common import float16
import torch
import os
import numpy as np
from datasets import Dataset, disable_progress_bar, enable_progress_bar
import logging


logger = logging.getLogger(__name__)


class SpanMarkerEncoderDummyInputenerator:
    SUPPORTED_INPUT_NAMES = [
        "input_ids",
        "attention_mask",
        "position_ids",
    ]
    BATCH_SIZE = 1

    @classmethod
    def generate_dummy_input(cls, vocab_size: int, sequence_length: int, torch_dtype: torch.dtype = torch.int32):
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
        }

        for value in cls.SUPPORTED_INPUT_NAMES:
            min_val = values[value]["min"]
            max_value = values[value]["max"]
            shape = values[value]["shape"]
            dummy_input[value] = torch.randint(low=min_val, high=max_value, size=shape, dtype=torch_dtype)
        return dummy_input


class SpanMarkerOnnx:
    """
    A class for performing named entity recognition using the SpanMarker model in ONNX format.

    This class handles the loading of ONNX models for both the encoder and classifier components of the SpanMarker model,
    manages data processing, and provides methods for making predictions on input data.

    >>> # Initialize a SpanMarkerOnnx
    >>> onnx_model = SpanMarkerOnnx(
        onnx_encoder_path="spanmarker_encoder.onnx",
        onnx_classifier_path="spanmarker_classifier.onnx",
        tokenizer=spanmarker_tokenizer,
        config=config,
        )

    After the model is loaded (and finetuned if it wasn't already), it can be used to predict entities:

    >>> onnx_model.predict("A prototype was fitted in the mid-'60s in a one-off DB5 extended 4'' after the doors and "
    ... "driven by Marek personally, and a normally 6-cylinder Aston Martin DB7 was equipped with a V8 unit in 1998.")
    [{'span': 'DB5', 'label': 'product-car', 'score': 0.8675689101219177, 'char_start_index': 52, 'char_end_index': 55},
     {'span': 'Marek', 'label': 'person-other', 'score': 0.9100819230079651, 'char_start_index': 99, 'char_end_index': 104},
     {'span': 'Aston Martin DB7', 'label': 'product-car', 'score': 0.9931442737579346, 'char_start_index': 143, 'char_end_index': 159}]
    """

    INPUT_TYPES = Union[str, List[str], List[List[str]], Dataset]
    OUTPUT_TYPES = Union[List[Dict[str, Union[str, int, float]]], List[List[Dict[str, Union[str, int, float]]]]]

    def __init__(
        self,
        onnx_encoder_path: Union[str, os.PathLike, pathlib.Path],
        onnx_classifier_path: Union[str, os.PathLike, pathlib.Path],
        config: SpanMarkerConfig,
        tokenizer: SpanMarkerTokenizer,
        show_progress_bar: bool = False,
        onnx_sess_options: SessionOptions = None,
        quantized: bool = False,
        providers: list = ["CPUExecutionProvider"],
    ):
        self.quantized = quantized
        self.show_progress_bar = show_progress_bar
        self.config = config
        self.tokenizer = tokenizer
        self.data_collator = SpanMarkerDataCollator(
            tokenizer=self.tokenizer, marker_max_length=self.config.marker_max_length
        )

        self.ort_encoder = self.load_ort_session(onnx_encoder_path, sess_options=onnx_sess_options, providers=providers)
        self.ort_classifier = self.load_ort_session(
            onnx_classifier_path, sess_options=onnx_sess_options, providers=providers
        )

        if torch.cuda.is_available() and providers[0] == "CUDAExecutionProvider":
            self.device = torch.device(device="cuda")
        elif torch.backends.mps.is_available() and providers[0] == "CoreMLExecutionProvider":
            self.device = torch.device(device="mps")
        else:
            self.device = torch.device(device="cpu")

    def load_ort_session(
        self, onnx_path: Union[str, os.PathLike], sess_options=None, providers=["CPUExecutionProvider"]
    ) -> ort.InferenceSession:
        """
        Loads an ONNX Runtime (ORT) inference session from a specified ONNX model path.
        This function initializes an ONNX Runtime inference session with the provided model. It offers customization through session options and execution providers.

        Args:
            onnx_path (Union[str, os.PathLike]):
                The file path to the ONNX model.
            sess_options (Optional[ort.SessionOptions]):
                Configuration options for the session. If not provided, default options are used with optimized settings for execution mode, graph optimization level, and the number of threads based on the CPU count.
            providers (List[str], optional):
                The execution providers to use. Defaults to ["CPUExecutionProvider"].

        Returns:
            ort.InferenceSession:
                An ONNX Runtime inference session configured with the specified model and session options.

        The function sets up default session options if none are provided, optimizing for sequential execution, enabling all graph optimizations, and setting the number of intra-operation threads to the number of CPU cores available. It then creates and returns an ORT inference session using these settings.
        """

        if not sess_options:
            sess_options = ort.SessionOptions()
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort_session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        return ort_session

    def data_to_device(self, data: torch.Tensor, classifier=False) -> None:
        if classifier:
            np_type = np.float16 if self.quantized else np.float32
            return data.detach().cpu().numpy().astype(np_type)
        return data.detach().cpu().numpy().astype(np.int32)


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        start_marker_indices: torch.Tensor,
        num_marker_pairs: torch.Tensor,
        num_words: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        sentence_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Moving the inputs to the device with onnx encoder
        onnx_input = {
            "input_ids": self.data_to_device(input_ids),
            "attention_mask": self.data_to_device(attention_mask),
            "position_ids": self.data_to_device(position_ids),
        }

        onnx_output = self.ort_encoder.run(None, input_feed=onnx_input)
        last_hidden_state = torch.from_numpy(onnx_output[0])
        sequence_length = last_hidden_state.size(1)
        batch_size = last_hidden_state.size(0)
        # Get the indices where the end markers start
        end_marker_indices = start_marker_indices + num_marker_pairs
        sequence_length_last_hidden_state = last_hidden_state.size(2) * 2

        # TODO: Solve the dynamic slicing problem when exporting with torchdynamo
        #  Pre-allocates the necessary space for feature_vector
        feature_vector = torch.zeros(batch_size, sequence_length // 2, sequence_length_last_hidden_state, device="cpu")
        for i in range(batch_size):
            feature_vector[
                i, : end_marker_indices[i] - start_marker_indices[i], : last_hidden_state.shape[-1]
            ] = last_hidden_state[i, start_marker_indices[i] : end_marker_indices[i]]
            feature_vector[
                i, : end_marker_indices[i] - start_marker_indices[i], last_hidden_state.shape[-1] :
            ] = last_hidden_state[i, end_marker_indices[i] : end_marker_indices[i] + num_marker_pairs[i]]

        # Moving the feature_vector to the device with the onnx classifier
        input_onnx_classifier = {"input": self.data_to_device(feature_vector, classifier=True)}
        logits = self.ort_classifier.run(None, input_feed=input_onnx_classifier)
        logits = torch.from_numpy(logits[0])

        return SpanMarkerOutput(
            logits=logits,
            out_num_marker_pairs=num_marker_pairs,
            out_num_words=num_words,
            out_document_ids=document_ids,
            out_sentence_ids=sentence_ids,
        )

    def predict(
        self,
        inputs: INPUT_TYPES,
        batch_size: int = 4,
        show_progress_bar: bool = False,
    ) -> OUTPUT_TYPES:
        """Predict named entities from input texts.

        Example::

            >>> model = SpanMarkerOnnx(...)
            >>> model.predict("Amelia Earhart flew her single engine Lockheed Vega 5B across the Atlantic to Paris.")
            [{'span': 'Amelia Earhart', 'label': 'person-other', 'score': 0.7629689574241638, 'char_start_index': 0, 'char_end_index': 14},
            {'span': 'Lockheed Vega 5B', 'label': 'product-airplane', 'score': 0.9833564758300781, 'char_start_index': 38, 'char_end_index': 54},
            {'span': 'Atlantic', 'label': 'location-bodiesofwater', 'score': 0.7621214389801025, 'char_start_index': 66, 'char_end_index': 74},
            {'span': 'Paris', 'label': 'location-GPE', 'score': 0.9807717204093933, 'char_start_index': 78, 'char_end_index': 83}]
            >>> model.predict(['Caesar', 'led', 'the', 'Roman', 'armies', 'in', 'the', 'Gallic', 'Wars', 'before', 'defeating', 'his', 'political', 'rival', 'Pompey', 'in', 'a', 'civil', 'war'])
            [{'span': ['Caesar'], 'label': 'person-politician', 'score': 0.683479905128479, 'word_start_index': 0, 'word_end_index': 1},
            {'span': ['Roman'], 'label': 'location-GPE', 'score': 0.7114525437355042, 'word_start_index': 3, 'word_end_index': 4},
            {'span': ['Gallic', 'Wars'], 'label': 'event-attack/battle/war/militaryconflict', 'score': 0.9015670418739319, 'word_start_index': 7, 'word_end_index': 9},
            {'span': ['Pompey'], 'label': 'person-politician', 'score': 0.9601260423660278, 'word_start_index': 14, 'word_end_index': 15}]

        Args:
            inputs (Union[str, List[str], List[List[str]], Dataset]): Input sentences from which to extract entities.
                Valid datastructures are:

                * str: a string sentence.
                * List[str]: a pre-tokenized string sentence, i.e. a list of words.
                * List[str]: a list of multiple string sentences.
                * List[List[str]]: a list of multiple pre-tokenized string sentences, i.e. a list with lists of words.
                * Dataset: A 🤗 :class:`~datasets.Dataset` with a ``tokens`` column and optionally ``document_id`` and ``sentence_id`` columns.
                    If the optional columns are provided, they will be used to provide document-level context.

            batch_size (int): The number of samples to include in a batch, a higher batch size is faster,
                but requires more memory. Defaults to 4
            show_progress_bar (bool): Whether to show a progress bar, useful for longer inputs. Defaults to `False`.

        Returns:
            Union[List[Dict[str, Union[str, int, float]]], List[List[Dict[str, Union[str, int, float]]]]]:
                If the input is a single sentence, then we output a list of dictionaries. Each dictionary
                represents one predicted entity, and contains the following keys:

                * ``label``: The predicted entity label.
                * ``span``: The text that the model deems an entity.
                * ``score``: The model its confidence.
                * ``word_start_index`` & ``word_end_index``: The word indices for the start/end of the entity,
                if the input is pre-tokenized.
                * ``char_start_index`` & ``char_end_index``: The character indices for the start/end of the entity,
                if the input is a string.

                If the input is multiple sentences, then we return a list containing multiple of the aforementioned lists.
        """
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
                show_progress_bar=show_progress_bar,
            )
            dataset = dataset.sort(column_names=["__sort_id"])
            dataset = dataset.remove_columns("__sort_id")
        elif self.config.trained_with_document_context:
            logger.warning(
                "This model was trained with document-level context: "
                "inference without document-level context may cause decreased performance."
            )

        if not show_progress_bar:
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
        if not show_progress_bar:
            enable_progress_bar()
        for batch_start_idx in trange(0, len(dataset), batch_size, leave=True, disable=not show_progress_bar):
            batch = dataset.select(range(batch_start_idx, min(len(dataset), batch_start_idx + batch_size)))
            # Expanding the small tokenized output into full-scale input_ids, position_ids and attention_mask matrices.
            batch = self.data_collator(batch)
            # Moving the inputs to the right device
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.no_grad():
                output = self.forward(**batch)
            # Computing probabilities based on the logits
            if self.quantized:
                probs = output.logits.float().softmax(-1)
            else:
                probs = output.logits.softmax(-1)
            # Get the labels and the correponding probability scores
            scores, labels = probs.max(-1)
            # TODO: Iterate over output.num_marker_pairs instead with enumerate
            for iter_idx in range(output.out_num_marker_pairs.size(0)):
                input_id = dataset["id"][batch_start_idx + iter_idx]
                out_num_marker_pairs = output.out_num_marker_pairs[iter_idx]
                results[input_id]["scores"].extend(scores[iter_idx, :out_num_marker_pairs].tolist())
                results[input_id]["labels"].extend(labels[iter_idx, :out_num_marker_pairs].tolist())
                results[input_id]["num_words"] = output.out_num_words[iter_idx]

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


def export_spanmarker_to_onnx(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    output_folder: Union[str, os.PathLike] = "spanmarker_onnx",
    opset_version: int = 13,
    device: str = "cpu",
    quantized: bool = False,
) -> None:
    """
    Exports a pretrained SpanMarker model to ONNX format.

    Args:
        pretrained_model_name_or_path (Union[str, os.PathLike]):
            The path or name of the pretrained SpanMarker model.
        output_folder (Union[str, os.PathLike]):
            The directory where the ONNX files will be saved. Defaults to 'spanmarker_onnx'.
        opset_version (int):
            The ONNX opset version to use for the model export. Defaults to 13.
        device (str):
            The device to use for model inference ('cpu' or 'gpu'). Defaults to 'cpu'.
        quantized (bool):
            If True, the exported model will be quantized for reduced model size and potentially faster inference.

    The function initializes the SpanMarker model components (encoder and classifier), generates dummy inputs,
    and then exports these components to ONNX format. If quantization is enabled, the model is further converted
    to a quantized version, which uses 16-bit floating-point numbers instead of 32-bit, reducing the model size and
    potentially improving performance on compatible hardware.

    Returns:
        None: This function does not return any value.
    """

    os.makedirs(output_folder, exist_ok=True)
    onnx_encoder_path = f"{output_folder}/spanmarker_encoder.onnx"
    onnx_classifier_path = f"{output_folder}/spanmarker_classifier.onnx"

    base_model = SpanMarkerModel.from_pretrained(pretrained_model_name_or_path)
    config = SpanMarkerConfig.from_pretrained(pretrained_model_name_or_path)
    encoder = base_model.encoder.eval()
    classifier = base_model.classifier.eval()

    # Dummy input for encoder and classifier
    encoder_dummy_input = SpanMarkerEncoderDummyInputenerator.generate_dummy_input(
        vocab_size=config.vocab_size, sequence_length=config.model_max_length_default
    )
    classifier_dummy_input = torch.randn(1, 256, 1536)

    # Moving to device
    encoder_dummy_input = {key: value.to(device=torch.device(device)) for key, value in encoder_dummy_input.items()}
    classifier_dummy_input = classifier_dummy_input.to(device=torch.device(device))
    encoder = encoder.to(device=torch.device(device))
    classifier = classifier.to(device=torch.device(device))

    # Export Onnx classifier
    torch.onnx.export(
        classifier,
        classifier_dummy_input,
        onnx_classifier_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Export Onnx encoder
    torch.onnx.export(
        encoder,
        encoder_dummy_input,
        onnx_encoder_path,
        input_names=[
            "input_ids",
            "attention_mask",
            "position_ids",
        ],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "position_ids": {0: "batch_size"},
            "last_hidden_state": {0: "batch_size"},
            "pooler_output": {0: "batch_size"},
        },
        do_constant_folding=True,
        export_params=True,
        opset_version=opset_version,
    )

    if quantized:
        for onnx_file in os.listdir(output_folder):
            model = onnx.load(os.path.join(output_folder, onnx_file))
            model_fp16 = float16.convert_float_to_float16(model)
            onnx_fp16_file = f"spanmarker_fp16_{onnx_file.split('_')[-1]}"
            onnx.save(model_fp16, os.path.join(output_folder, onnx_fp16_file))
