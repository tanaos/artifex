import pytest

from artifex import Artifex


@pytest.mark.integration
def test_train_multi_label_model(
    artifex: Artifex,
    output_folder: str
):
    """
    Test training a MultiLabelClassificationModel through a concrete implementation.
    This test uses LLMOutputGuardrail as a concrete implementation of MultiLabelClassificationModel.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
        output_folder (str): Temporary folder for saving training outputs.
    """
    
    # Use LLMOutputGuardrail as a concrete implementation
    model = artifex.llm_output_guardrail
    
    labels = {
        "category1": "First category description",
        "category2": "Second category description",
        "category3": "Third category description"
    }
    
    model.train(
        unsafe_categories=labels,
        num_samples=40,
        num_epochs=1,
        output_path=output_folder,
        device=-1,
        language="english",
        disable_logging=True
    )
    
    # Verify model configuration
    assert model._model.config.problem_type == "multi_label_classification"
    assert len(model._label_names) == 3
    assert set(model._label_names) == set(labels.keys())


@pytest.mark.integration
def test_inference_multi_label_model(
    artifex: Artifex
):
    """
    Test inference with a MultiLabelClassificationModel through a concrete implementation.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    from artifex.core import MultiLabelClassificationResponse
    
    # Use LLMOutputGuardrail as a concrete implementation
    model = artifex.llm_output_guardrail
    
    results = model("Test input text", device=-1, disable_logging=True)
    
    # Verify response structure
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], MultiLabelClassificationResponse)
    assert isinstance(results[0].labels, dict)
    
    # Verify all probabilities are valid
    for prob in results[0].labels.values():
        assert 0 <= prob <= 1


@pytest.mark.integration
def test_batch_inference_multi_label_model(
    artifex: Artifex
):
    """
    Test batch inference with a MultiLabelClassificationModel.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    from artifex.core import MultiLabelClassificationResponse
    
    # Use LLMOutputGuardrail as a concrete implementation
    model = artifex.llm_output_guardrail
    
    inputs = ["First text", "Second text", "Third text"]
    results = model(inputs, device=-1, disable_logging=True)
    
    # Verify response structure
    assert isinstance(results, list)
    assert len(results) == 3
    assert all(isinstance(r, MultiLabelClassificationResponse) for r in results)
    
    # Verify all probabilities are valid
    for result in results:
        assert isinstance(result.labels, dict)
        for prob in result.labels.values():
            assert 0 <= prob <= 1


@pytest.mark.integration
def test_load_multi_label_model(
    artifex: Artifex,
    output_folder: str
):
    """
    Test loading a trained MultiLabelClassificationModel.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
        output_folder (str): Temporary folder for saving training outputs.
    """
    
    # Train and save a model
    model1 = artifex.llm_output_guardrail
    
    labels = {
        "cat1": "Category 1",
        "cat2": "Category 2"
    }
    
    model1.train(
        unsafe_categories=labels,
        num_samples=40,
        num_epochs=1,
        output_path=output_folder,
        device=-1,
        disable_logging=True
    )
    
    # Create a new model instance and load the saved model
    from artifex.models.classification.multi_label_classification import LLMOutputGuardrail
    model2 = LLMOutputGuardrail(synthex=artifex._synthex_client)
    model2.load(f"{output_folder}/output_model")
    
    # Verify the loaded model has the correct configuration
    assert model2._model.config.problem_type == "multi_label_classification"
    assert set(model2._label_names) == set(labels.keys())
    
    # Verify the loaded model can perform inference
    results = model2("Test input", device=-1, disable_logging=True)
    assert len(results) == 1
    assert isinstance(results[0].labels, dict)
