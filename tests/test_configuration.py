import re

import pytest
from transformers import AutoConfig

from span_marker.modeling import SpanMarkerConfig, SpanMarkerModel
from tests.constants import CONLL_LABELS, TINY_BERT


def test_config_without_O() -> None:
    with pytest.raises(ValueError, match=re.escape("'O' label")):
        SpanMarkerModel.from_pretrained(TINY_BERT, labels=["person", "location", "misc"])


def test_config_get_default(finetuned_conll_span_marker_model: SpanMarkerModel) -> None:
    config = finetuned_conll_span_marker_model.config
    # Verify that we can fetch directly from top-level SpanMarkerConfig
    assert config.get("marker_max_length") == 128
    # Verify that we can fetch from the underlying encoder, even with multiple options
    assert config.get("hidden_dropout_prob") == 0.1
    assert config.get(["dropout_rate", "hidden_dropout_prob"]) == 0.1
    # Verify that we get the default if the config value does not exist
    assert config.get(["does_not_exist", "also_does_not_exist"], default=12) == 12


def test_config_with_schemed_labels() -> None:
    encoder_config = AutoConfig.from_pretrained(TINY_BERT)
    encoder_config.id2label = dict(enumerate(CONLL_LABELS))
    encoder_config.label2id = {v: k for k, v in encoder_config.id2label.items()}
    config = SpanMarkerConfig(encoder_config=encoder_config.to_dict())
    assert config.id2reduced_id == {0: 0, 1: 4, 2: 4, 3: 3, 4: 3, 5: 1, 6: 1, 7: 2, 8: 2}
    assert config.label2id == {"LOC": 1, "MISC": 2, "O": 0, "ORG": 3, "PER": 4}
    assert config.id2label == {0: "O", 1: "LOC", 2: "MISC", 3: "ORG", 4: "PER"}
    assert config.outside_id == 0
