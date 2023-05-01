from pathlib import Path

from span_marker.model_card import generate_model_card
from span_marker.modeling import SpanMarkerModel


def test_model_card(finetuned_fewnerd_span_marker_model: SpanMarkerModel, tmp_path: Path) -> None:
    config = finetuned_fewnerd_span_marker_model.config
    model_card = generate_model_card(tmp_path, config)
    assert (
        "uses [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny) as the underlying encoder" in model_card
    )
    assert f'SpanMarkerModel.from_pretrained("span_marker_model_name")' in model_card
    assert "\n\n\n" not in model_card
    assert "\n\n## Usage" in model_card

    config.encoder["_name_or_path"] = "does_not_exist"
    model_card = generate_model_card(tmp_path, config)
    assert 'uses "does_not_exist" as the underlying encoder' in model_card
    assert "\n\n\n" not in model_card
    assert "\n\n## Usage" in model_card

    del config.encoder["_name_or_path"]
    model_card = generate_model_card(tmp_path, config)
    assert "as the underlying encoder" not in model_card
    assert "\n\n\n" not in model_card
    assert "\n\n## Usage" in model_card

    model_card = generate_model_card("tomaarsen/my_test_model", config)
    assert f'SpanMarkerModel.from_pretrained("tomaarsen/my_test_model")' in model_card
    assert "\n\n\n" not in model_card
    assert "\n\n## Usage" in model_card
