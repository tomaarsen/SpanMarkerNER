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

## [Unreleased]

### Fixed

- Resolved tricky issue causing models to learn to never predict the last token as an entity (Closes [#1](https://github.com/tomaarsen/SpanMarkerNER/pull/1)).

## [0.2.2] - 2023-04-13

### Fixed

- Correctly propagate `SpanMarkerModel.from_pretrained` kwargs to Config initialisation.

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