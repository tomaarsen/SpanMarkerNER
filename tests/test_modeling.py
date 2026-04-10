import logging
import re
from typing import Dict, List, Optional, Union

import pytest
import torch
from datasets import Dataset

from span_marker.configuration import SpanMarkerConfig
from span_marker.modeling import SpanMarkerModel
from span_marker.tokenizer import SpanMarkerTokenizer
from tests.constants import CONLL_LABELS, FEWNERD_COARSE_LABELS, TINY_BERT
from tests.helpers import compare_entities


@pytest.mark.parametrize(
    ("model_name", "labels"),
    [
        (TINY_BERT, CONLL_LABELS),
        (TINY_BERT, FEWNERD_COARSE_LABELS),
        (TINY_BERT, None),
        ("tomaarsen/span-marker-bert-tiny-conll03", CONLL_LABELS),
        ("tomaarsen/span-marker-bert-tiny-conll03", None),
        ("tomaarsen/span-marker-bert-tiny-fewnerd-coarse-super", FEWNERD_COARSE_LABELS),
        ("tomaarsen/span-marker-bert-tiny-fewnerd-coarse-super", None),
    ],
)
def test_from_pretrained(model_name: str, labels: Optional[List[str]]) -> None:
    def load() -> SpanMarkerModel:
        return SpanMarkerModel.from_pretrained(model_name, labels=labels).try_cuda()

    # If labels have to be provided
    if model_name == TINY_BERT and labels is None:
        with pytest.raises(
            ValueError, match=re.escape("Please provide a `labels` list to `SpanMarkerModel.from_pretrained()`")
        ):
            model = load()
        return

    # Otherwise, we should always be able to call from_pretrained
    model = load()
    assert isinstance(model, SpanMarkerModel)
    assert isinstance(model.config, SpanMarkerConfig)
    assert isinstance(model.tokenizer, SpanMarkerTokenizer)

    for sentence in [
        "This might just output confusing things like M.C. Escher, but it should at least not crash in Germany.",
        ["This might just output confusing things like M.C. Escher, but it should at least not crash in Germany."],
        "This might just output confusing things like M.C. Escher , but it should at least not crash in Germany .".split(),
        [
            "This might just output confusing things like M.C. Escher , but it should at least not crash in Germany .".split()
        ],
    ]:
        output = model.predict(sentence)
        assert isinstance(output, list)
    output = model.predict([])
    assert output == []


@pytest.mark.parametrize(
    ("inputs", "gold_entities"),
    [
        (
            "I'm living in the Netherlands, but I work in Spain.",
            [
                {"span": "Netherlands", "label": "LOC", "char_start_index": 18, "char_end_index": 29},
                {"span": "Spain", "label": "LOC", "char_start_index": 45, "char_end_index": 50},
            ],
        ),
        (
            "I'm living in the Netherlands , but I work in Spain .".split(),
            [
                {"span": ["Netherlands"], "label": "LOC", "word_start_index": 4, "word_end_index": 5},
                {"span": ["Spain"], "label": "LOC", "word_start_index": 10, "word_end_index": 11},
            ],
        ),
    ],
)
def test_correct_predictions(
    finetuned_conll_span_marker_model: SpanMarkerModel,
    inputs: Union[str, List[str]],
    gold_entities: List[Dict[str, Union[str, int]]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = finetuned_conll_span_marker_model.try_cuda()

    # Single sentence
    pred_entities = model.predict(inputs)
    compare_entities(pred_entities, gold_entities)

    # Single sentence, but nested
    pred_entity_list = model.predict([inputs])
    for pred_entities in pred_entity_list:
        compare_entities(pred_entities, gold_entities)

    # Multiple sentences
    pred_entity_list = model.predict([inputs] * 3)
    for pred_entities in pred_entity_list:
        compare_entities(pred_entities, gold_entities)

    # As a Dataset
    pred_entity_list = model.predict(Dataset.from_dict({"tokens": [inputs]}))
    for pred_entities in pred_entity_list:
        compare_entities(pred_entities, gold_entities)

    # As a non-singular Dataset
    pred_entity_list = model.predict(Dataset.from_dict({"tokens": [inputs] * 2}))
    for pred_entities in pred_entity_list:
        compare_entities(pred_entities, gold_entities)

    # Check for warning if the model is trained with document-level context
    caplog.clear()
    model.config.trained_with_document_context = True
    model.predict(inputs)
    assert any(
        [
            level == logging.WARNING
            and text == "This model was trained with document-level context: "
            "inference without document-level context may cause decreased performance."
            for (_, level, text) in caplog.record_tuples
        ]
    )


@pytest.mark.parametrize(
    ("inputs", "gold_entity_list"),
    [
        (
            Dataset.from_dict(
                {
                    "tokens": [
                        "I'm living in the Netherlands, but I work in Spain.",
                        "My name is Tom and this is a test.",
                        "I hope it can detect Paris here.",
                        "And nothing in this sentence.",
                    ],
                    "document_id": [0, 0, 0, 0],
                    "sentence_id": [0, 1, 2, 3],
                }
            ),
            [
                [
                    {"span": "Netherlands", "label": "LOC", "char_start_index": 18, "char_end_index": 29, "document_id": 0, "sentence_id": 0},
                    {"span": "Spain", "label": "LOC", "char_start_index": 45, "char_end_index": 50, "document_id": 0, "sentence_id": 0},
                ],
                [{"span": "Tom", "label": "PER", "char_start_index": 11, "char_end_index": 14, "document_id": 0, "sentence_id": 1}],
                [{"span": "Paris", "label": "LOC", "char_start_index": 21, "char_end_index": 26, "document_id": 0, "sentence_id": 2}],
                [],
            ],
        ),
        (
            Dataset.from_dict(
                {
                    "tokens": [
                        "I'm living in the Netherlands, but I work in Spain.",
                        "My name is Tom and this is a test.",
                        "I hope it can detect Paris here.",
                        "And nothing in this sentence.",
                    ],
                    "document_id": [0, 1, 0, 0],
                    "sentence_id": [0, 0, 2, 1],
                }
            ),
            [
                [
                    {"span": "Netherlands", "label": "LOC", "char_start_index": 18, "char_end_index": 29, "document_id": 0, "sentence_id": 0},
                    {"span": "Spain", "label": "LOC", "char_start_index": 45, "char_end_index": 50, "document_id": 0, "sentence_id": 0},
                ],
                [{"span": "Tom", "label": "PER", "char_start_index": 11, "char_end_index": 14, "document_id": 1, "sentence_id": 0}],
                [{"span": "Paris", "label": "LOC", "char_start_index": 21, "char_end_index": 26, "document_id": 0, "sentence_id": 2}],
                [],
            ],
        ),
    ],
)
def test_correct_predictions_with_document_level_context(
    finetuned_conll_span_marker_model: SpanMarkerModel,
    inputs: Union[str, List[str]],
    gold_entity_list: List[Dict[str, Union[str, int]]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = finetuned_conll_span_marker_model.try_cuda()

    pred_entity_list = model.predict(inputs)
    for pred_entities, gold_entities in zip(pred_entity_list, gold_entity_list):
        compare_entities(pred_entities, gold_entities)

    assert any(
        [
            level == logging.WARNING
            and text == "This model was trained without document-level context: "
            "inference with document-level context may cause decreased performance."
            for (_, level, text) in caplog.record_tuples
        ]
    )


def test_predict_where_first_sentence_is_word(finetuned_conll_span_marker_model: SpanMarkerModel) -> None:
    model = finetuned_conll_span_marker_model.try_cuda()
    outputs = model.predict(["One", "Two Three Four Five"])
    assert len(outputs) == 2
    assert isinstance(outputs[0], list)


def test_predict_empty_error(finetuned_conll_span_marker_model: SpanMarkerModel) -> None:
    model = finetuned_conll_span_marker_model.try_cuda()
    with pytest.raises(ValueError, match="The `SpanMarkerTokenizer` detected an empty sentence, please remove it."):
        model.predict(["One Two", "Three Four Five", ""])


def test_incorrect_predict_inputs(finetuned_conll_span_marker_model: SpanMarkerModel):
    model = finetuned_conll_span_marker_model.try_cuda()
    with pytest.raises(ValueError, match="could not recognize your input"):
        model.predict(12)
    with pytest.raises(ValueError, match="could not recognize your input"):
        model.predict(True)


@pytest.mark.parametrize(
    "kwargs",
    [
        # Reasonable kwargs that will be used by SpanMarkerConfig
        {
            "model_max_length": 256,
            "marker_max_length": 128,
            "entity_max_length": 8,
        },
        # Kwargs that will be used by from_pretrained of the Encoder
        {"low_cpu_mem_usage": True},
        # Completely arbitrary kwargs that should be discarded/ignored
        {"this_is_completely_unused_I_hope": True},
    ],
)
def test_load_with_kwargs(kwargs) -> None:
    # We only test that the model can be loaded without issues
    model = SpanMarkerModel.from_pretrained(TINY_BERT, labels=CONLL_LABELS, kwargs=kwargs)
    assert isinstance(model, SpanMarkerModel)


def test_try_cuda(finetuned_conll_span_marker_model: SpanMarkerModel) -> None:
    # This should not crash, regardless of whether Torch is compiled with CUDA or not
    finetuned_conll_span_marker_model.try_cuda()
    # The model is on CUDA if CUDA is available, and not on CUDA if CUDA is not available.
    assert (finetuned_conll_span_marker_model.device.type == "cuda") == torch.cuda.is_available()
