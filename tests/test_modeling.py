import re
from typing import Dict, List, Optional, Union

import pytest

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
        return SpanMarkerModel.from_pretrained(model_name, labels=labels)

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
) -> None:
    model = finetuned_conll_span_marker_model

    # Single sentence
    pred_entities = model.predict(inputs)
    compare_entities(pred_entities, gold_entities)

    # Multiple sentences
    pred_entity_list = model.predict([inputs] * 3)
    for pred_entities in pred_entity_list:
        compare_entities(pred_entities, gold_entities)
