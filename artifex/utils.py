import os
from artifex.config import config


def get_model_output_path(output_path: str) -> str:
    """
    Get the output path for the trained model based on the provided output path (its parent directory).
    """

    return str(os.path.join(output_path, config.SYNTHEX_OUTPUT_MODEL_FOLDER_NAME))

def get_dataset_output_path(output_path: str) -> str:
    """
    Get the output path for the dataset based on the provided output path (its parent directory).
    """

    return str(os.path.join(output_path, config.DEFAULT_SYNTHEX_DATASET_NAME))
