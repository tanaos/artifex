import pytest
from typing import List
from artifex.models.classification import NClassClassificationModel
from synthex import Synthex


class DummyNClassClassificationModel(NClassClassificationModel):
    """
    Dummy concrete implementation for testing NClassClassificationModel.
    """
    
    @property
    def _base_model_name(self) -> str:
        return "dummy-model"

    @property
    def _system_data_gen_instr(self) -> List[str]:
        return [
            "System instruction 1 for {domain}",
            "System instruction 2 for {domain}"
        ]


@pytest.fixture
def model(mocker) -> DummyNClassClassificationModel:
    """
    Fixture that returns a DummyNClassClassificationModel instance with mocked Synthex.
    Args:
        mocker: The pytest-mock fixture for mocking.
    Returns:
        DummyNClassClassificationModel: An instance of the dummy model for testing.
    """
    
    synthex_mock = mocker.Mock(spec=Synthex)
    return DummyNClassClassificationModel(synthex=synthex_mock)


def test_get_data_gen_instr_basic(model: DummyNClassClassificationModel):
    """
    Test that _get_data_gen_instr correctly formats system instructions and combines them
    with user instructions, excluding the domain from user instructions.
    """

    user_instr = [
        "classA: descriptionA",
        "classB: descriptionB",
        "test-domain"
    ]
    result = model._get_data_gen_instr(user_instr)
    expected = [
        "System instruction 1 for test-domain",
        "System instruction 2 for test-domain",
        "classA: descriptionA",
        "classB: descriptionB"
    ]
    assert result == expected


def test_get_data_gen_instr_empty_classes(model: DummyNClassClassificationModel):
    """
    Test that _get_data_gen_instr works when there are no class instructions, only the domain.
    """

    user_instr = [
        "test-domain"
    ]
    result = model._get_data_gen_instr(user_instr)
    expected = [
        "System instruction 1 for test-domain",
        "System instruction 2 for test-domain"
    ]
    assert result == expected


def test_get_data_gen_instr_multiple_classes(model: DummyNClassClassificationModel):
    """
    Test that _get_data_gen_instr works with multiple class instructions.
    """

    user_instr = [
        "classA: descriptionA",
        "classB: descriptionB",
        "classC: descriptionC",
        "test-domain"
    ]
    result = model._get_data_gen_instr(user_instr)
    expected = [
        "System instruction 1 for test-domain",
        "System instruction 2 for test-domain",
        "classA: descriptionA",
        "classB: descriptionB",
        "classC: descriptionC"
    ]
    assert result == expected

  
def test_get_data_gen_instr_domain_with_spaces(model: DummyNClassClassificationModel):
    """
    Test that _get_data_gen_instr correctly formats instructions when the domain contains spaces.
    """

    user_instr = [
        "classA: descriptionA",
        "classB: descriptionB",
        "complex domain name"
    ]
    result = model._get_data_gen_instr(user_instr)
    expected = [
        "System instruction 1 for complex domain name",
        "System instruction 2 for complex domain name",
        "classA: descriptionA",
        "classB: descriptionB"
    ]
    assert result == expected


def test_get_data_gen_instr_no_classes_only_domain(model: DummyNClassClassificationModel):
    """
    Test that _get_data_gen_instr returns only system instructions when user_instr contains only the domain.
    """

    user_instr = ["domain-only"]
    result = model._get_data_gen_instr(user_instr)
    expected = [
        "System instruction 1 for domain-only",
        "System instruction 2 for domain-only"
    ]
    assert result == expected


def test_get_data_gen_instr_empty_user_instr(model: DummyNClassClassificationModel):
    """
    Test that _get_data_gen_instr raises IndexError when user_instr is empty.
    """

    with pytest.raises(IndexError):
        model._get_data_gen_instr([])


def test_get_data_gen_instr_special_characters(model: DummyNClassClassificationModel):
    """
    Test that _get_data_gen_instr works with special characters in class descriptions and domain.
    """

    user_instr = [
        "classA: descr!ption@A#",
        "classB: descr$ption%B^",
        "domain*&^%$#@!"
    ]
    result = model._get_data_gen_instr(user_instr)
    expected = [
        "System instruction 1 for domain*&^%$#@!",
        "System instruction 2 for domain*&^%$#@!",
        "classA: descr!ption@A#",
        "classB: descr$ption%B^"
    ]
    assert result == expected