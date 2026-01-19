## Release v0.6.0 - January 19, 2026

### Added

- Added observability features for inference process, including CPU and RAM usage tracking, input and output logging and token count metrics.
- Added observability features for training process, including CPU and RAM usage tracking, epoch duration and loss metrics.
- Added monitoring feature to track potential issues during inference, including low confidence score warnings.

### Fixed

- Fixed security vulnerabilities by updating dependencies.

## Release v0.5.1 - January 9, 2026

### Added

- Add entity type to masking token in the `TextAnonymization.__call__` method.

### Fixed

- Fixed security vulnerabilities by updating dependencies.

## Release v0.5.0 - January 2, 2026

### Added

- Added the `SpamDetection` model.
- Added the `TopicClassification` model.
- Added explicit support for model language selection through the `language` parameter in the `train` method.
- Added support for device selection in the `train` method.

## Release v0.4.1 - December 22, 2025

### Changed

- Turned `ClassificationModel` into a concrete class instead of an abstract class.
- Replaced `instructions` argument with `unsafe_content` argument in the `Guardrail.train()` 
method.

### Fixed

- Fixed security vulnerabilities by updating dependencies.
- Suppressed annoying tokenization-related warning in `NamedEntityRecognition.__call__()` method.

## Release v0.4.0 - December 4, 2025

### Added

- Added the `Reranker` model.
- Added the `SentimentAnalysis` model.
- Added the `EmotionDetection` model.
- Added the `NamedEntityRecognition` model.
- Added the `TextAnonymization` model.
- Added integration tests.

### Fixed

- Fixed a bug causing the "Generating training data" progress bar to display a wrong progress percentage.

### Removed

- Removed support for Python <= 3.9.
- Removed all `# type: ignore` comments from the codebase.

### Changed

- Updated error message when the `.load()` method is provided with a nonexistent model path or an invalid file format.
- Updated the `IntentClassifier` base model.
- Updated the output model's directory structure for better organization.

## Release v0.3.2 - October 2, 2025

### Added

- Added support for Python 3.09.

### Fixed

- Fixed SyntaxError in the `config.py` file, which was causing issues during library initialization on Python versions earlier than 3.13.

## Release v0.3.1 - September 16, 2025

### Added

- Added a spinner on library load.
- Added a spinner on model load.
- Added a log message containing the generated model path, once model generation is complete.

### Changed

- Replaced data generation `tqdm` progress bar with `rich` progress bar.
- Replaced data processing `tqdm` progress bar with a `rich` spinner.
- Replaced model training `tqdm` progress bar with `rich` progress bar.
- Moved the output model to a dedicated `output_model` directory within the output directory.

### Removed

- Removed all intermediate model checkpoints.
- Removed unnecessary output files from output directory.

## Release v0.3.0 - September 9, 2025

First release