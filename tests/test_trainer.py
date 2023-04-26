from pathlib import Path
from typing import Dict, List

import pytest
from datasets import Dataset, DatasetDict
from transformers import EvalPrediction

from span_marker.modeling import SpanMarkerModel
from span_marker.trainer import Trainer
from tests.constants import CONLL_LABELS, DEFAULT_ARGS, TINY_BERT


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
def test_trainer_standard(
    model_fixture: str, dataset_fixture: str, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    model: SpanMarkerModel = request.getfixturevalue(model_fixture)
    dataset: DatasetDict = request.getfixturevalue(dataset_fixture)

    # Perform training and evaluation
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

    # Try saving and loading the model
    model_path = tmp_path / model_fixture / dataset_fixture
    model.save_pretrained(model_path)
    loaded_model = model.from_pretrained(model_path)
    output = loaded_model.predict(
        "This might just output confusing things like M.C. Escher, but it should at least not crash in Germany."
    )
    assert isinstance(output, list)


def test_trainer_model_init(
    finetuned_conll_span_marker_model: SpanMarkerModel, conll_dataset_dict: DatasetDict
) -> None:
    model = finetuned_conll_span_marker_model
    dataset = conll_dataset_dict

    def model_init() -> SpanMarkerModel:
        return model

    trainer = Trainer(
        model_init=model_init, args=DEFAULT_ARGS, train_dataset=dataset["train"], eval_dataset=dataset["test"]
    )
    trainer.train()
    metrics = trainer.evaluate()
    assert isinstance(metrics, dict)
    output = trainer.model.predict(
        "This might just output confusing things like M.C. Escher, but it should at least not crash in Germany."
    )
    assert isinstance(output, list)


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


@pytest.mark.parametrize("column_names", [["tokens", "tags"], ["text", "ner_tags"], ["id", "text", "labels"]])
def test_trainer_incorrect_columns(finetuned_conll_span_marker_model: SpanMarkerModel, column_names: List[str]) -> None:
    model = finetuned_conll_span_marker_model
    dataset = Dataset.from_dict({column_name: [] for column_name in column_names})

    trainer = Trainer(model, args=DEFAULT_ARGS, train_dataset=dataset, eval_dataset=dataset)
    with pytest.raises(ValueError, match="The train dataset must contain a '.*?' column."):
        trainer.train()

    with pytest.raises(ValueError, match="The evaluation dataset must contain a '.*?' column."):
        trainer.evaluate()


def test_trainer_entity_tracker_warning(conll_dataset_dict: DatasetDict, caplog) -> None:
    model = SpanMarkerModel.from_pretrained(TINY_BERT, labels=CONLL_LABELS, entity_max_length=1)
    trainer = Trainer(
        model, args=DEFAULT_ARGS, train_dataset=conll_dataset_dict["train"], eval_dataset=conll_dataset_dict["train"]
    )
    trainer.train()
    assert any(["model will ignore" in record.msg for record in caplog.records])
    trainer.evaluate()
    assert any(["model won't be able to predict" in record.msg for record in caplog.records])
