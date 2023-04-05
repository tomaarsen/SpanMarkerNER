from typing import Dict

import pytest
from datasets import DatasetDict
from transformers import EvalPrediction

from span_marker.modeling import SpanMarkerModel
from span_marker.trainer import Trainer
from tests.constants import DEFAULT_ARGS


@pytest.mark.parametrize(
    ("model_fixture", "dataset_fixture"),
    [
        ("fresh_conll_span_marker_model", "conll_dataset_dict"),  # IOB2
        ("finetuned_conll_span_marker_model", "conll_dataset_dict"),  # IOB2
        ("fresh_fewnerd_span_marker_model", "fewnwerd_coarse_dataset_dict"),  # no scheme
        ("finetuned_fewnerd_span_marker_model", "fewnwerd_coarse_dataset_dict"),  # no scheme
        ("fresh_fabner_span_marker_model", "fabner_dataset_dict"),  # BIOES
    ],
)
def test_trainer_standard(model_fixture: str, dataset_fixture: str, request: pytest.FixtureRequest) -> None:
    model: SpanMarkerModel = request.getfixturevalue(model_fixture)
    dataset: DatasetDict = request.getfixturevalue(dataset_fixture)

    trainer = Trainer(model, args=DEFAULT_ARGS, train_dataset=dataset["train"], eval_dataset=dataset["test"])
    trainer.train()
    metrics = trainer.evaluate()
    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == {
        "eval_loss",
        "eval_overall_f1",
        "eval_overall_recall",
        "eval_overall_precision",
        "eval_overall_accuracy",
        "eval_runtime",
        "eval_samples_per_second",
        "eval_steps_per_second",
        "epoch",
    }


def test_trainer_compute_metrics(
    finetuned_conll_span_marker_model: SpanMarkerModel, conll_dataset_dict: DatasetDict
) -> None:
    model = finetuned_conll_span_marker_model
    eval_dataset = conll_dataset_dict["test"]

    def compute_metrics(eval_prediction: EvalPrediction) -> Dict[str, float]:
        return {"custom_metric": 0.74}

    trainer = Trainer(model, args=DEFAULT_ARGS, eval_dataset=eval_dataset, compute_metrics=compute_metrics)
    metrics = trainer.evaluate()
    assert "eval_custom_metric" in metrics.keys()
