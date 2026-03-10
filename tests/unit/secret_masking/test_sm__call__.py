import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from unittest.mock import ANY, mock_open, patch

from artifex.models import SecretMasking
from artifex.config import config


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    return mocker.Mock(spec=Synthex)


@pytest.fixture
def secret_masking(mock_synthex: Synthex, mocker: MockerFixture) -> SecretMasking:
    mocker.patch.object(SecretMasking.__bases__[0], '__init__', return_value=None)
    instance = SecretMasking(mock_synthex)
    return instance


@pytest.mark.unit
def test_call_single_file_no_entities_detected(
    secret_masking: SecretMasking, mocker: MockerFixture
):
    """
    Tests __call__ with a single file path when no secrets are detected.
    The returned content should be identical to the original file content.
    """

    file_content = 'import os\n\nprint("hello world")\n'

    # file has 2 non-empty lines: 'import os' and 'print("hello world")'
    mocker.patch.object(
        SecretMasking.__bases__[0], '__call__', return_value=[[], []]
    )

    with patch("builtins.open", mock_open(read_data=file_content)):
        result = secret_masking("script.py")

    assert result == [file_content]


@pytest.mark.unit
def test_call_single_file_with_api_key_entity(
    secret_masking: SecretMasking, mocker: MockerFixture
):
    """
    Tests __call__ with a single file where an API_KEY entity is detected on one line.
    The entity value should be replaced with the default mask token.
    """

    file_content = 'API_KEY = "sk-abc123xyz"\n'

    # 'API_KEY = "sk-abc123xyz"' -> "sk-abc123xyz" starts at 10, ends at 24 (len=24)
    mock_entity = mocker.Mock()
    mock_entity.entity_group = "API_KEY"
    mock_entity.start = 10
    mock_entity.end = 24

    mocker.patch.object(
        SecretMasking.__bases__[0], '__call__', return_value=[[mock_entity]]
    )

    with patch("builtins.open", mock_open(read_data=file_content)):
        result = secret_masking("script.py")

    expected_mask = config.DEFAULT_SECRET_MASKING_MASK
    expected_line = f'API_KEY = {expected_mask}\n'
    assert result == [expected_line]


@pytest.mark.unit
def test_call_list_of_files(
    secret_masking: SecretMasking, mocker: MockerFixture
):
    """
    Tests __call__ with a list of file paths, one secret per file.
    """

    content1 = 'PASSWORD = "s3cr3t!"\n'
    content2 = 'DB_URL = "postgres://user:pass@host/db"\n'

    mock_entity1 = mocker.Mock()
    mock_entity1.entity_group = "PASSWORD"
    mock_entity1.start = 11
    mock_entity1.end = 20

    # 'DB_URL = "postgres://user:pass@host/db"' -> quoted URL starts at 9, len=39
    mock_entity2 = mocker.Mock()
    mock_entity2.entity_group = "DB_CONNECTION"
    mock_entity2.start = 9
    mock_entity2.end = 39

    mock_parent = mocker.patch.object(
        SecretMasking.__bases__[0], '__call__',
        side_effect=[[[mock_entity1]], [[mock_entity2]]]
    )

    expected_mask = config.DEFAULT_SECRET_MASKING_MASK

    def _open_side_effect(path, *args, **kwargs):
        data = content1 if path == "file1.py" else content2
        return mock_open(read_data=data)()

    with patch("builtins.open", side_effect=_open_side_effect):
        result = secret_masking(["file1.py", "file2.py"])

    assert result == [
        f'PASSWORD = {expected_mask}\n',
        f'DB_URL = {expected_mask}\n',
    ]


@pytest.mark.unit
def test_call_with_include_mask_type(
    secret_masking: SecretMasking, mocker: MockerFixture
):
    """
    Tests __call__ with include_mask_type=True. The mask token should include the entity type.
    """

    file_content = 'SECRET_KEY = "mysecret"\n'

    mock_entity = mocker.Mock()
    mock_entity.entity_group = "SECRET_KEY"
    mock_entity.start = 13
    mock_entity.end = 23

    mocker.patch.object(
        SecretMasking.__bases__[0], '__call__', return_value=[[mock_entity]]
    )

    with patch("builtins.open", mock_open(read_data=file_content)):
        result = secret_masking("script.py", include_mask_type=True)

    assert result == ['SECRET_KEY = [REDACTED_SECRET_KEY]\n']


@pytest.mark.unit
def test_call_with_include_mask_type_closing_bracket(
    secret_masking: SecretMasking, mocker: MockerFixture
):
    """
    Tests __call__ with include_mask_type=True and a mask token with a closing bracket.
    The entity type should be inserted before the closing bracket.
    """

    file_content = 'TOKEN = "abc"\n'

    mock_entity = mocker.Mock()
    mock_entity.entity_group = "ACCESS_TOKEN"
    mock_entity.start = 8
    mock_entity.end = 13

    mocker.patch.object(
        SecretMasking.__bases__[0], '__call__', return_value=[[mock_entity]]
    )

    with patch("builtins.open", mock_open(read_data=file_content)):
        result = secret_masking("script.py", mask_token="[REDACTED]", include_mask_type=True)

    assert result == ['TOKEN = [REDACTED_ACCESS_TOKEN]\n']


@pytest.mark.unit
def test_call_invalid_entity_raises_value_error(
    secret_masking: SecretMasking, mocker: MockerFixture
):
    """
    Tests that passing an unknown entity in entities_to_mask raises ValueError.
    """

    with patch("builtins.open", mock_open(read_data="x = 1\n")):
        with pytest.raises(ValueError, match="cannot be masked"):
            secret_masking("script.py", entities_to_mask=["UNKNOWN_ENTITY"])


@pytest.mark.unit
def test_call_entities_to_mask_filters_correctly(
    secret_masking: SecretMasking, mocker: MockerFixture
):
    """
    Tests that only entities listed in entities_to_mask are masked; others are left as-is.
    """

    file_content = 'API_KEY = "key123"\nPASSWORD = "pass456"\n'

    mock_api_entity = mocker.Mock()
    mock_api_entity.entity_group = "API_KEY"
    mock_api_entity.start = 10
    mock_api_entity.end = 18

    mock_pw_entity = mocker.Mock()
    mock_pw_entity.entity_group = "PASSWORD"
    mock_pw_entity.start = 11
    mock_pw_entity.end = 20

    mocker.patch.object(
        SecretMasking.__bases__[0], '__call__',
        return_value=[[mock_api_entity], [mock_pw_entity]]
    )

    expected_mask = config.DEFAULT_SECRET_MASKING_MASK

    with patch("builtins.open", mock_open(read_data=file_content)):
        result = secret_masking("script.py", entities_to_mask=["API_KEY"])

    # Only API_KEY line masked; PASSWORD line should be unchanged
    masked_lines = result[0].split('\n')
    assert expected_mask in masked_lines[0]
    assert expected_mask not in masked_lines[1]


@pytest.mark.unit
def test_call_empty_lines_are_preserved(
    secret_masking: SecretMasking, mocker: MockerFixture
):
    """
    Tests that empty lines in the file are preserved in the output.
    """

    file_content = 'import os\n\nAPI_KEY = "key"\n'

    # 'import os' is non-empty line 0; 'API_KEY = "key"' is non-empty line 1
    # 'API_KEY = "key"' -> "key" spans [10:15]
    mock_entity = mocker.Mock()
    mock_entity.entity_group = "API_KEY"
    mock_entity.start = 10
    mock_entity.end = 15

    # Two non-empty lines: 'import os' (no entities) and 'API_KEY = "key"'
    mocker.patch.object(
        SecretMasking.__bases__[0], '__call__', return_value=[[], [mock_entity]]
    )

    with patch("builtins.open", mock_open(read_data=file_content)):
        result = secret_masking("script.py")

    lines = result[0].split('\n')
    assert lines[0] == 'import os'
    assert lines[1] == ''


@pytest.mark.unit
def test_call_with_custom_mask_token(
    secret_masking: SecretMasking, mocker: MockerFixture
):
    """
    Tests __call__ with a custom mask token.
    """

    file_content = 'API_KEY = "sk-secret"\n'

    mock_entity = mocker.Mock()
    mock_entity.entity_group = "API_KEY"
    mock_entity.start = 10
    mock_entity.end = 21

    mocker.patch.object(
        SecretMasking.__bases__[0], '__call__', return_value=[[mock_entity]]
    )

    with patch("builtins.open", mock_open(read_data=file_content)):
        result = secret_masking("script.py", mask_token="***")

    assert result == ['API_KEY = ***\n']
