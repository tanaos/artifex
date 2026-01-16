import pytest
from pytest_mock import MockerFixture

from artifex.core.decorators.logging import _count_tokens


@pytest.mark.unit
def test_count_tokens_with_single_string(mocker: MockerFixture):
    """
    Test that _count_tokens correctly counts tokens for a single string.
    """
    mock_tokenizer = mocker.MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    
    result = _count_tokens("Hello world", mock_tokenizer)
    
    assert result == 5
    mock_tokenizer.encode.assert_called_once_with("Hello world", add_special_tokens=True)


@pytest.mark.unit
def test_count_tokens_with_list_of_strings(mocker: MockerFixture):
    """
    Test that _count_tokens correctly counts tokens for a list of strings.
    """
    mock_tokenizer = mocker.MagicMock()
    mock_tokenizer.encode.side_effect = [
        [1, 2, 3],      # First string: 3 tokens
        [4, 5, 6, 7],   # Second string: 4 tokens
        [8, 9]          # Third string: 2 tokens
    ]
    
    texts = ["Hello", "world", "!"]
    result = _count_tokens(texts, mock_tokenizer)
    
    assert result == 9  # 3 + 4 + 2
    assert mock_tokenizer.encode.call_count == 3


@pytest.mark.unit
def test_count_tokens_with_empty_string(mocker: MockerFixture):
    """
    Test that _count_tokens handles empty strings correctly.
    """
    mock_tokenizer = mocker.MagicMock()
    mock_tokenizer.encode.return_value = []
    
    result = _count_tokens("", mock_tokenizer)
    
    assert result == 0
    mock_tokenizer.encode.assert_called_once_with("", add_special_tokens=True)


@pytest.mark.unit
def test_count_tokens_with_empty_list(mocker: MockerFixture):
    """
    Test that _count_tokens handles empty list correctly.
    """
    mock_tokenizer = mocker.MagicMock()
    
    result = _count_tokens([], mock_tokenizer)
    
    assert result == 0
    mock_tokenizer.encode.assert_not_called()


@pytest.mark.unit
def test_count_tokens_with_list_of_empty_strings(mocker: MockerFixture):
    """
    Test that _count_tokens handles a list of empty strings correctly.
    """
    mock_tokenizer = mocker.MagicMock()
    mock_tokenizer.encode.side_effect = [[], [], []]
    
    result = _count_tokens(["", "", ""], mock_tokenizer)
    
    assert result == 0
    assert mock_tokenizer.encode.call_count == 3


@pytest.mark.unit
def test_count_tokens_with_long_text(mocker: MockerFixture):
    """
    Test that _count_tokens correctly handles long text with many tokens.
    """
    mock_tokenizer = mocker.MagicMock()
    # Simulate a long text with 100 tokens
    mock_tokenizer.encode.return_value = list(range(100))
    
    long_text = "This is a very long text " * 20
    result = _count_tokens(long_text, mock_tokenizer)
    
    assert result == 100
    mock_tokenizer.encode.assert_called_once_with(long_text, add_special_tokens=True)


@pytest.mark.unit
def test_count_tokens_with_single_item_list(mocker: MockerFixture):
    """
    Test that _count_tokens handles a list with a single string.
    """
    mock_tokenizer = mocker.MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4]
    
    result = _count_tokens(["Single text"], mock_tokenizer)
    
    assert result == 4
    mock_tokenizer.encode.assert_called_once_with("Single text", add_special_tokens=True)


@pytest.mark.unit
def test_count_tokens_with_special_characters(mocker: MockerFixture):
    """
    Test that _count_tokens handles text with special characters.
    """
    mock_tokenizer = mocker.MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6]
    
    special_text = "Hello! @#$%^&* 世界"
    result = _count_tokens(special_text, mock_tokenizer)
    
    assert result == 6
    mock_tokenizer.encode.assert_called_once_with(special_text, add_special_tokens=True)


@pytest.mark.unit
def test_count_tokens_converts_string_to_list(mocker: MockerFixture):
    """
    Test that _count_tokens converts a string input to a list internally.
    """
    mock_tokenizer = mocker.MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    
    # Pass a string
    result = _count_tokens("test", mock_tokenizer)
    
    assert result == 3
    # Should be called once (string converted to list internally)
    assert mock_tokenizer.encode.call_count == 1


@pytest.mark.unit
def test_count_tokens_with_varying_length_strings(mocker: MockerFixture):
    """
    Test that _count_tokens correctly sums tokens from strings of varying lengths.
    """
    mock_tokenizer = mocker.MagicMock()
    mock_tokenizer.encode.side_effect = [
        [1],                    # 1 token
        [1, 2, 3, 4, 5],       # 5 tokens
        [1, 2],                # 2 tokens
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 10 tokens
    ]
    
    texts = ["a", "short sentence", "hi", "this is a much longer sentence"]
    result = _count_tokens(texts, mock_tokenizer)
    
    assert result == 18  # 1 + 5 + 2 + 10
    assert mock_tokenizer.encode.call_count == 4


@pytest.mark.unit
def test_count_tokens_adds_special_tokens(mocker: MockerFixture):
    """
    Test that _count_tokens calls encode with add_special_tokens=True.
    """
    mock_tokenizer = mocker.MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    
    _count_tokens("test", mock_tokenizer)
    
    # Verify add_special_tokens=True is passed
    mock_tokenizer.encode.assert_called_with("test", add_special_tokens=True)


@pytest.mark.unit
def test_count_tokens_with_whitespace_only(mocker: MockerFixture):
    """
    Test that _count_tokens handles whitespace-only strings.
    """
    mock_tokenizer = mocker.MagicMock()
    mock_tokenizer.encode.return_value = [1]  # Tokenizers might encode whitespace
    
    result = _count_tokens("   ", mock_tokenizer)
    
    assert result == 1
    mock_tokenizer.encode.assert_called_once_with("   ", add_special_tokens=True)


@pytest.mark.unit
def test_count_tokens_with_newlines(mocker: MockerFixture):
    """
    Test that _count_tokens handles text with newlines.
    """
    mock_tokenizer = mocker.MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    
    text_with_newlines = "Line 1\nLine 2\nLine 3"
    result = _count_tokens(text_with_newlines, mock_tokenizer)
    
    assert result == 5
    mock_tokenizer.encode.assert_called_once_with(text_with_newlines, add_special_tokens=True)


@pytest.mark.unit
def test_count_tokens_with_mixed_list(mocker: MockerFixture):
    """
    Test that _count_tokens handles a list with various text types.
    """
    mock_tokenizer = mocker.MagicMock()
    mock_tokenizer.encode.side_effect = [
        [1, 2],          # "Hello"
        [3],             # Empty or short
        [4, 5, 6, 7],    # Longer text
        []               # Empty string
    ]
    
    texts = ["Hello", "", "This is longer", ""]
    result = _count_tokens(texts, mock_tokenizer)
    
    assert result == 7  # 2 + 1 + 4 + 0
    assert mock_tokenizer.encode.call_count == 4


@pytest.mark.unit
def test_count_tokens_accumulates_correctly(mocker: MockerFixture):
    """
    Test that _count_tokens correctly accumulates token counts across multiple strings.
    """
    mock_tokenizer = mocker.MagicMock()
    # Each string has exactly 3 tokens
    mock_tokenizer.encode.side_effect = [[1, 2, 3]] * 5
    
    texts = ["text1", "text2", "text3", "text4", "text5"]
    result = _count_tokens(texts, mock_tokenizer)
    
    assert result == 15  # 3 * 5
    assert mock_tokenizer.encode.call_count == 5
