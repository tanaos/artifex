import pytest


@pytest.fixture(autouse=True)
def mock_cognitor_globally(mocker):
    """
    Auto-mock cognitor.Cognitor for all unit tests so that model __call__ methods
    do not attempt real Cognitor initialisation or logging.
    Tests that specifically verify Cognitor behaviour should patch cognitor.Cognitor
    themselves with the appropriate mock setup, which will take precedence over this fixture.
    """
    mocker.patch("cognitor.Cognitor")
