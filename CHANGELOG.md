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