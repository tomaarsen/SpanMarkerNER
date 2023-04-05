import re

import pytest

from span_marker.modeling import SpanMarkerModel
from tests.constants import TINY_BERT


def test_config_without_O() -> None:
    with pytest.raises(ValueError, match=re.escape("'O' label")):
        SpanMarkerModel.from_pretrained(TINY_BERT, labels=["person", "location", "misc"])


def test_config_get_default(finetuned_conll_span_marker_model: SpanMarkerModel) -> None:
    config = finetuned_conll_span_marker_model.config
    # Verify that we can fetch directly from top-level SpanMarkerConfig
    assert config.get("marker_max_length") == 256
    # Verify that we can fetch from the underlying encoder, even with multiple options
    assert config.get("hidden_dropout_prob") == 0.1
    assert config.get(["dropout_rate", "hidden_dropout_prob"]) == 0.1
    # Verify that we get the default if the config value does not exist
    assert config.get(["does_not_exist", "also_does_not_exist"], default=12) == 12
