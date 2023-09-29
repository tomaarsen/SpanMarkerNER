from transformers import pipeline

import span_marker


def test_pipeline() -> None:
    pipe = pipeline(task="span-marker", model="tomaarsen/span-marker-bert-tiny-fewnerd-coarse-super")
    outputs = pipe("Tom lives in the Netherlands.")
    assert len(outputs) == 2
    assert outputs[0]["span"] == "Tom"
    assert outputs[1]["span"] == "Netherlands"
