import pytest

from span_marker.modeling import SpanMarkerModel
from tests.constants import CONLL_LABELS


@pytest.fixture
def fresh_span_marker_model() -> SpanMarkerModel:
    return SpanMarkerModel.from_pretrained("prajjwal1/bert-tiny", labels=CONLL_LABELS)


@pytest.fixture
def span_marker_model() -> SpanMarkerModel:
    return SpanMarkerModel.from_pretrained("tomaarsen/span-marker-bert-tiny-conll03")
