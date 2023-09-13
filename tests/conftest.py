import os

import datasets
import pytest
from datasets import DatasetDict

from span_marker.modeling import SpanMarkerModel
from tests.constants import (
    CONLL_LABELS,
    FABNER_LABELS,
    FEWNERD_COARSE_LABELS,
    TINY_BERT,
)


def pytest_sessionstart(session) -> None:
    # Disable caching (for tests only) to ensure that we're actually recomputing things
    datasets.disable_caching()


@pytest.fixture(scope="function", autouse=True)
def randomize_seed() -> None:
    # Pytest sets the random seed the same for all runs. Combined with disabled caching, this
    # causes the "random" caching files to not be random at all, causing conflicts on Windows.
    # This makes tests fail when the code actually works.

    # So, we import the instance overseeing the temporary folder used when caching is disabled,
    # and we clean it up after every test. We then have to re-initialize it and make a fresh
    # directory in its place, otherwise the errors persist.

    yield

    from datasets.fingerprint import (
        _TEMP_DIR_FOR_TEMP_CACHE_FILES,
        get_temporary_cache_files_directory,
    )

    if _TEMP_DIR_FOR_TEMP_CACHE_FILES:
        _TEMP_DIR_FOR_TEMP_CACHE_FILES._cleanup()
        os.mkdir(_TEMP_DIR_FOR_TEMP_CACHE_FILES.name)


# CoNLL03
@pytest.fixture(scope="session")
def conll_dataset_dict() -> DatasetDict:
    return DatasetDict.load_from_disk("tests/data/tiny_conll2003")


@pytest.fixture(scope="session")
def document_context_conll_dataset_dict() -> DatasetDict:
    return DatasetDict.load_from_disk("tests/data/tiny_conll2003_with_context")


@pytest.fixture()
def fresh_conll_span_marker_model() -> SpanMarkerModel:
    return SpanMarkerModel.from_pretrained(TINY_BERT, labels=CONLL_LABELS)


@pytest.fixture()
def finetuned_conll_span_marker_model() -> SpanMarkerModel:
    return SpanMarkerModel.from_pretrained("tomaarsen/span-marker-bert-tiny-conll03")


# FewNERD Supervised
@pytest.fixture(scope="session")
def fewnwerd_coarse_dataset_dict() -> DatasetDict:
    return DatasetDict.load_from_disk("tests/data/tiny_fewnerd_super").remove_columns("fine_ner_tags")


@pytest.fixture()
def fresh_fewnerd_span_marker_model() -> SpanMarkerModel:
    return SpanMarkerModel.from_pretrained(TINY_BERT, labels=FEWNERD_COARSE_LABELS)


@pytest.fixture()
def finetuned_fewnerd_span_marker_model() -> SpanMarkerModel:
    return SpanMarkerModel.from_pretrained("tomaarsen/span-marker-bert-tiny-fewnerd-coarse-super")


# FabNER
@pytest.fixture(scope="session")
def fabner_dataset_dict() -> DatasetDict:
    return DatasetDict.load_from_disk("tests/data/tiny_fabner")


@pytest.fixture()
def fresh_fabner_span_marker_model() -> SpanMarkerModel:
    return SpanMarkerModel.from_pretrained(TINY_BERT, labels=FABNER_LABELS)
