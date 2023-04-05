import re
from typing import List, Optional

import pytest

from span_marker.configuration import SpanMarkerConfig
from span_marker.modeling import SpanMarkerModel
from span_marker.tokenizer import SpanMarkerTokenizer
from tests.constants import CONLL_LABELS, FEWNERD_COARSE_LABELS, TINY_BERT


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

    output = model.predict(
        "This might just output confusing things like M.C. Escher, but it should at least not crash in Germany."
    )
    assert isinstance(output, list)
