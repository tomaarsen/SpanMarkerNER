import logging
import re
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
        ("fresh_conll_span_marker_model", "document_context_conll_dataset_dict"),  # IOB2, doc-context
        ("finetuned_conll_span_marker_model", "document_context_conll_dataset_dict"),  # IOB2, doc-context
        ("fresh_fewnerd_span_marker_model", "fewnwerd_coarse_dataset_dict"),  # no scheme
        ("finetuned_fewnerd_span_marker_model", "fewnwerd_coarse_dataset_dict"),  # no scheme
        ("fresh_fabner_span_marker_model", "fabner_dataset_dict"),  # BIOES
    ],
)
def test_trainer_standard(
    model_fixture: str,
    dataset_fixture: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    model: SpanMarkerModel = request.getfixturevalue(model_fixture)
    dataset: DatasetDict = request.getfixturevalue(dataset_fixture)

    # Perform training and evaluation
    trainer = Trainer(model, args=DEFAULT_ARGS, train_dataset=dataset["train"], eval_dataset=dataset["test"])
    trainer.train()
    if "document_context" in dataset_fixture:
        assert model.config.trained_with_document_context
    metrics = trainer.evaluate()
    assert isinstance(metrics, dict)
    labels = {label for label, _id in model.config.label2id.items() if _id != model.config.outside_id}
    keys = {f"eval_{label}" for label in labels}
    assert set(metrics.keys()) <= {
        "eval_loss",
        "eval_overall_f1",
        "eval_overall_recall",
        "eval_overall_precision",
        "eval_overall_accuracy",
        "eval_runtime",
        "eval_samples_per_second",
        "eval_steps_per_second",
        "epoch",
        *keys,
    }
    for key in keys:
        if key in metrics:
            assert metrics[key].keys() == {"f1", "number", "precision", "recall"}

    # Try saving and loading the model
    model_path = tmp_path / model_fixture / dataset_fixture
    model.save_pretrained(model_path)
    loaded_model = model.from_pretrained(model_path).try_cuda()
    output = loaded_model.predict(
        "This might just output confusing things like M.C. Escher, but it should at least not crash in Germany."
    )
    assert isinstance(output, list)

    if "document_context" in dataset_fixture:
        # If there's document context, let's evaluate the doc-context model again, but with just tokens
        caplog.clear()
        trainer.evaluate(dataset["test"].remove_columns(("document_id", "sentence_id")))
        assert any(
            [
                level == logging.WARNING
                and text == "This model was trained with document-level context: "
                "evaluation without document-level context may cause decreased performance."
                for (_, level, text) in caplog.record_tuples
            ]
        )
        # Alternatively, let's pretend the model is not trained with doc-level context,
        # and use the doc-level dataset for evaluation
        caplog.clear()
        model.config.trained_with_document_context = False
        trainer.evaluate()
        assert any(
            [
                level == logging.WARNING
                and text == "This model was trained without document-level context: "
                "evaluation with document-level context may cause decreased performance."
                for (_, level, text) in caplog.record_tuples
            ]
        )


@pytest.mark.parametrize(
    "dataset_fixture",
    [
        "conll_dataset_dict",
        "document_context_conll_dataset_dict",
    ],
)
def test_trainer_model_init(
    finetuned_conll_span_marker_model: SpanMarkerModel, dataset_fixture: str, request: pytest.FixtureRequest
) -> None:
    model = finetuned_conll_span_marker_model
    dataset: DatasetDict = request.getfixturevalue(dataset_fixture)

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


def test_trainer_entity_tracker_warning_entity_length(conll_dataset_dict: DatasetDict, caplog) -> None:
    model = SpanMarkerModel.from_pretrained(TINY_BERT, labels=CONLL_LABELS, entity_max_length=1)
    trainer = Trainer(
        model, args=DEFAULT_ARGS, train_dataset=conll_dataset_dict["train"], eval_dataset=conll_dataset_dict["train"]
    )
    trainer.train()
    train_pattern = re.compile(
        r"This SpanMarker model will ignore [\d\.]+% of all annotated entities in the train dataset\. "
        r"This is caused by the SpanMarkerModel maximum entity length of 1 word\.\n"
        r"These are the frequencies of the missed entities due to maximum entity length out of \d+ total entities:"
    )
    assert any([train_pattern.search(record.msg) for record in caplog.records])
    assert any(["Detected the IOB or IOB2 labeling scheme." in record.msg for record in caplog.records])
    trainer.evaluate()
    eval_pattern = re.compile(
        r"This SpanMarker model won't be able to predict [\d\.]+% of all annotated entities in the evaluation dataset\. "
        r"This is caused by the SpanMarkerModel maximum entity length of 1 word\.\n"
        r"These are the frequencies of the missed entities due to maximum entity length out of \d+ total entities:"
    )
    assert any([eval_pattern.search(record.msg) for record in caplog.records])


def test_trainer_entity_tracker_warning_model_length(conll_dataset_dict: DatasetDict, caplog) -> None:
    model = SpanMarkerModel.from_pretrained(TINY_BERT, labels=CONLL_LABELS, model_max_length=5)
    trainer = Trainer(
        model, args=DEFAULT_ARGS, train_dataset=conll_dataset_dict["train"], eval_dataset=conll_dataset_dict["train"]
    )
    trainer.train()
    train_pattern = re.compile(
        r"This SpanMarker model will ignore [\d\.]+% of all annotated entities in the train dataset\. "
        r"This is caused by the SpanMarkerModel maximum model input length of 5 tokens\.\n"
        r"A total of \d+ \([\d\.]+%\) entities were missed due to the maximum input length\."
    )
    assert any([train_pattern.match(record.msg) for record in caplog.records])
    assert any(["Detected the IOB or IOB2 labeling scheme." in record.msg for record in caplog.records])
    trainer.evaluate()
    eval_pattern = re.compile(
        r"This SpanMarker model won't be able to predict [\d\.]+% of all annotated entities in the evaluation dataset\. "
        r"This is caused by the SpanMarkerModel maximum model input length of 5 tokens\.\n"
        r"A total of \d+ \([\d\.]+%\) entities were missed due to the maximum input length\."
    )
    assert any([eval_pattern.match(record.msg) for record in caplog.records])


def test_trainer_entity_tracker_warning_entity_and_model_length(conll_dataset_dict: DatasetDict, caplog) -> None:
    model = SpanMarkerModel.from_pretrained(TINY_BERT, labels=CONLL_LABELS, model_max_length=5, entity_max_length=1)
    trainer = Trainer(
        model, args=DEFAULT_ARGS, train_dataset=conll_dataset_dict["train"], eval_dataset=conll_dataset_dict["train"]
    )
    trainer.train()
    train_pattern = re.compile(
        r"This SpanMarker model will ignore [\d\.]+% of all annotated entities in the train dataset\. "
        r"This is caused by the SpanMarkerModel maximum entity length of 1 word and the maximum model "
        r"input length of 5 tokens\.\n"
        r"These are the frequencies of the missed entities due to maximum entity length out of \d+ total entities:\n"
        r".*\nAdditionally, a total of \d+ \([\d\.]+%\) entities were missed due to the maximum input length\."
    )
    assert any([train_pattern.match(record.msg) for record in caplog.records])
    assert any(["Detected the IOB or IOB2 labeling scheme." in record.msg for record in caplog.records])
    trainer.evaluate()
    eval_pattern = re.compile(
        r"This SpanMarker model won't be able to predict [\d\.]+% of all annotated entities in the evaluation dataset\. "
        r"This is caused by the SpanMarkerModel maximum entity length of 1 word and the maximum model "
        r"input length of 5 tokens\.\n"
        r"These are the frequencies of the missed entities due to maximum entity length out of \d+ total entities:\n"
        r".*\nAdditionally, a total of \d+ \([\d\.]+%\) entities were missed due to the maximum input length\."
    )
    assert any([eval_pattern.match(record.msg) for record in caplog.records])
