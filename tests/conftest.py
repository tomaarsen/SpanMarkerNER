import pytest

from span_marker.modeling import SpanMarkerModel

CONLL_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


@pytest.fixture
def fresh_span_marker_model() -> SpanMarkerModel:
    return SpanMarkerModel.from_pretrained("prajjwal1/bert-tiny", labels=CONLL_LABELS)


@pytest.fixture
def span_marker_model() -> SpanMarkerModel:
    return SpanMarkerModel.from_pretrained("tomaarsen/span-marker-bert-tiny-conll03")
