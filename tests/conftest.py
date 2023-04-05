import pytest
from datasets import DatasetDict, load_dataset

from span_marker.modeling import SpanMarkerModel
from tests.constants import CONLL_LABELS, FEWNERD_COARSE_LABELS


@pytest.fixture()
def fresh_conll_span_marker_model() -> SpanMarkerModel:
    return SpanMarkerModel.from_pretrained("prajjwal1/bert-tiny", labels=CONLL_LABELS)


@pytest.fixture()
def finetuned_conll_span_marker_model() -> SpanMarkerModel:
    return SpanMarkerModel.from_pretrained("tomaarsen/span-marker-bert-tiny-conll03")


@pytest.fixture()
def fresh_fewnerd_span_marker_model() -> SpanMarkerModel:
    return SpanMarkerModel.from_pretrained("prajjwal1/bert-tiny", labels=FEWNERD_COARSE_LABELS)


@pytest.fixture()
def finetuned_fewnerd_span_marker_model() -> SpanMarkerModel:
    return SpanMarkerModel.from_pretrained("tomaarsen/span-marker-bert-tiny-fewnerd-coarse-super")


@pytest.fixture(scope="session")
def conll_dataset_dict() -> DatasetDict:
    train_ds, test_ds = load_dataset("conll2003", split=["train[:2]", "test[:10]"])
    return DatasetDict({"train": train_ds, "test": test_ds})


@pytest.fixture(scope="session")
def fewnwerd_coarse_dataset_dict() -> DatasetDict:
    train_ds, test_ds = load_dataset("DFKI-SLT/few-nerd", "supervised", split=["train[:2]", "test[:10]"])
    return DatasetDict({"train": train_ds, "test": test_ds})


@pytest.fixture(scope="session")
def fewnwerd_fine_dataset_dict() -> DatasetDict:
    train_ds, test_ds = load_dataset("DFKI-SLT/few-nerd", "supervised", split=["train[:2]", "test[:10]"])
    dataset = DatasetDict({"train": train_ds, "test": test_ds})
    dataset = dataset.remove_columns("ner_tags")
    return dataset.rename_column("fine_ner_tags", "ner_tags")
