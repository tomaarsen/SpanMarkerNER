# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
Types of changes
* "Added" for new features.
* "Changed" for changes in existing functionality.
* "Deprecated" for soon-to-be removed features.
* "Removed" for now removed features.
* "Fixed" for any bug fixes.
* "Security" in case of vulnerabilities.
-->

## [1.4.0]

### Added

- Added `SpanMarkerModel.generate_model_card()` method to get a model card string.
- Added `SpanMarkerModelCardData` that should be passed to `SpanMarkerModel.from_pretrained` with additional information like
  - `language`, `license`, `model_name`, `model_id`, `encoder_name`, `encoder_id`, `dataset_name`, `dataset_id`, `dataset_revision`.
- Added `transformers` `pipeline` support, e.g. `pipeline(task="span-marker", model="tomaarsen/span-marker-mbert-base-multinerd")`.

### Changed

- Heavily improved automatic model card generated.
- Evaluating outside of training now returns per-label outputs instead of only "overall" F1, precision and recall.
- Warn if the used tokenizer distinguishes between punctuation directly attached to a word and punctuation separated from a word by a space.
  - If so, then inference of that model will require the punctuation to be split from the words.
- Improve label normalization speed.
- Allow you to call SpanMarkerModel.from_pretrained with a pre-initialized SpanMarkerConfig.

### Deprecated

- Deprecated Python 3.7.

### Fixed

- Fixed tokenization mismatch between training and inference for XLM-RoBERTa models: allows for normal inference of those models.
- Resolve niche bug when TrainingArguments are not provided.

## [1.3.0]

### Added

- Added an `overwrite_entities` parameter to the spaCy pipeline component to allow for overwriting spaCy entities.
- Added `.pipe()` method to spaCy integration to allow for batched inference.

### Changed

- Stop overwriting spaCy entities by default.

## [1.2.5]

### Fixed

- Allow for immutable `TrainingArguments` from newer `transformers` release.

## [1.2.4]

### Fixed

- Resolved broken license information.

## [1.2.3]

### Fixed

- Fix crash in spaCy inference when using subsequent whitespace.

## [1.2.2]

### Added

- Added support for using `span_marker` spaCy pipeline component without importing SpanMarker.

## [1.2.1]

### Added

- Added support for `load_in_8bit=True` and `device_map="auto"`.

## [1.2.0]

### Added

- Added `trained_with_document_context` to the SpanMarkerConfig.
  - Added warnings if a model is trained with document-context and evaluated/inferenced without, or vice versa.
- Added `spaCy` integration via `nlp.add_pipe("span_marker")`. See the [SpanMarker with spaCy documentation](https://tomaarsen.github.io/SpanMarkerNER/notebooks/spacy_integration.html) for information.

### Changed

- Heavily improved computational efficiency of sample spreading, resulting in notably faster inference speeds.
- Disable progress bar for inference by default, and add `show_progress_bar` parameter to `SpanMarkerModel.predict`.

### Fixed

- Fixed evaluation method failing when the testing dataset contains two adjacent and identical sentences.

## [1.1.1]

### Fixed

- Add missing space in model card template.
- Return nested list if input is a singular list of sentences or a dataset with one sample.

## [1.1.0]

### Added

- Added support for document-level context in training, evaluation and inference.
  - Use it by supplying `document_id` and `sentence_id` columns to the Trainer datasets.
  - Tune it by supplying `max_prev_context` and `max_next_context` to the `SpanMarkerConfig` via `SpanMarkerModel.from_pretrained(..., max_prev_context=3)`.
- Added batch inference support via `SpanMarkerModel.predict(..., batch_size=4)`.

### Changed

- Ensure models are in evaluation mode when using `SpanMarkerModel.predict`.

### Deprecated

- Removed the `allow_overlapping` optional keyword from `SpanMarkerModel.predict`

## [1.0.1]

### Fixed

- Fixed critical issue with incorrect predictions at inputs that require multiple samples.

## [1.0.0]

### Added

- Added a warning for entities that are ignored/skipped due to the maximum entity length or maximum model input length.
- Added info-level logs displaying the detected labeling scheme (IOB/IOB2, BIOES, BILOU, none).
- Added a warning suggesting to use `model.cuda()` when predictions are performed on a CPU while CUDA is available.
- Added `try_cuda` method to `SpanMarkerModel` which tries to place the model on CUDA and does nothing if that fails.

### Changed

- Updated where in the input IDs the span markers are stored, results in 40% training and inferencing speed increase.
- Updated default `marker_max_length` in SpanMarkerConfig from 256 to 128.
- Updated default `entity_max_length` in SpanMarkerConfig from 16 to 8.
- Add support for `datasets<2.6.0`.
- Add warning if a `<v1.0.0` model is loaded using `v1.0.0` or newer.
- Propagate `SpanMarkerModel.from_pretrained` kwargs to the encoder its `AutoModel.from_pretrained`.
- Ignore `UndefinedMetricWarning` when evaluation f1 is 0.
- Improved model card generation.

### Fixed

- Resolved tricky issue causing models to learn to never predict the last token as an entity (Closes [#1](https://github.com/tomaarsen/SpanMarkerNER/pull/1)).
- Fixed label normalization for BILOU datasets.

## [0.2.2] - 2023-04-13

### Fixed

- Correctly propagate `SpanMarkerModel.from_pretrained` kwargs to the config initialisation.

## [0.2.1] - 2023-04-07

### Added

- Save `span_marker_version` in config files from now on.

### Changed

- `SpanMarkerModel.save_pretrained` and `SpanMarkerModel.push_to_hub` now also pushes the tokenizer and a simple model card.

## [0.2.0] - 2023-04-06

### Added

- Added missing docstrings.

### Changed

- Updated how entity span indices are returned for `SpanMarkerModel.predict`.

### Fixed

- Prevent incorrect labels when loading a model trained with a schemed (e.g. IOB, BIOES) dataset.
- Fix several bugs with loading finetuned SpanMarker models.
- Add missing methods to `SpanMarkerTokenizer`.
- Fix endless recursion bug when providing a `compute_metrics` to the Trainer.

## [0.1.1] - 2023-03-31

### Fixed

- Prevent crash when `args` not supplied to Trainer.
- Prevent crash on evaluation when using `fp16=True` as a Training Argument.

## [0.1.0] - 2023-03-30

### Added

- Implement initial working version.