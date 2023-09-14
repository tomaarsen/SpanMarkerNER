import logging
from pathlib import Path

import pytest
from datasets import DatasetDict

from span_marker import (
    SpanMarkerModel,
    SpanMarkerModelCardData,
    Trainer,
    TrainingArguments,
)
from span_marker.model_card import generate_model_card, is_on_huggingface

from .constants import FEWNERD_COARSE_LABELS
from .model_card_pattern import MODEL_CARD_PATTERN


def test_model_card(fewnwerd_coarse_dataset_dict: DatasetDict, tmp_path: Path) -> None:
    base_encoder_id = "prajjwal1/bert-tiny"
    model = SpanMarkerModel.from_pretrained(
        base_encoder_id,
        labels=FEWNERD_COARSE_LABELS,
        model_card_data=SpanMarkerModelCardData(
            model_id="tomaarsen/span-marker-test-model-card",
            dataset_id="conll2003",
            dataset_name="CoNLL 2003",
            encoder_id=base_encoder_id,
            language="en",
            license="apache-2.0",
        ),
    )
    train_dataset = fewnwerd_coarse_dataset_dict["train"]
    eval_dataset = fewnwerd_coarse_dataset_dict["test"].select(range(1))

    args = TrainingArguments(
        str(tmp_path),
        report_to="codecarbon",
        eval_steps=1,
        per_device_train_batch_size=1,
        evaluation_strategy="steps",
        num_train_epochs=1,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    model_card = generate_model_card(trainer.model)
    assert MODEL_CARD_PATTERN.fullmatch(model_card)


def test_model_card_languages() -> None:
    base_encoder_id = "prajjwal1/bert-tiny"
    model = SpanMarkerModel.from_pretrained(
        base_encoder_id,
        labels=FEWNERD_COARSE_LABELS,
        model_card_data=SpanMarkerModelCardData(
            language=["en", "nl", "de"],
        ),
    )
    model_card = model.generate_model_card()
    assert "**Languages:** en, nl, de" in model_card


def test_model_card_warnings(caplog: pytest.LogCaptureFixture):
    SpanMarkerModelCardData(dataset_id="test_value")
    assert any(
        [
            level == logging.WARNING
            and text == "The provided 'test_value' dataset could not be found on the Hugging Face Hub."
            " Setting `dataset_id` to None."
            for (_, level, text) in caplog.record_tuples
        ]
    )

    caplog.clear()
    SpanMarkerModelCardData(encoder_id="test_value")
    assert any(
        [
            level == logging.WARNING
            and text == "The provided 'test_value' model could not be found on the Hugging Face Hub."
            " Setting `encoder_id` to None."
            for (_, level, text) in caplog.record_tuples
        ]
    )

    caplog.clear()
    SpanMarkerModelCardData(model_id="test_value")
    assert any(
        [
            level == logging.WARNING
            and text == "The provided 'test_value' model ID should include the organization or user,"
            ' such as "tomaarsen/span-marker-mbert-base-multinerd". Setting `model_id` to None.'
            for (_, level, text) in caplog.record_tuples
        ]
    )


def test_is_on_huggingface_edge_case() -> None:
    assert not is_on_huggingface("test_value")
