import pytest

from artifex import Artifex
from artifex.core import ClassificationResponse


THREAT_LABELS = ["no_threat", "brute_force", "sql_injection", "ddos", "malware", "unauthorized_access"]


@pytest.mark.integration
def test__call__single_input_success(artifex: Artifex):
    """
    Test the `__call__` method of ThreatDetection with a single log entry.
    Ensure that:
    - It returns a list of ClassificationResponse objects.
    - The output label is one of the expected threat labels.
    """
    out = artifex.threat_detection()(
        "192.168.1.5 - - [03/Apr/2026:10:15:00 +0000] GET /?id=1 UNION SELECT 1,2,3-- HTTP/1.1 200",
        device=-1, disable_logging=True
    )
    assert isinstance(out, list)
    assert all(isinstance(resp, ClassificationResponse) for resp in out)
    assert all(resp.label in THREAT_LABELS for resp in out)


@pytest.mark.integration
def test__call__multiple_inputs_success(artifex: Artifex):
    """
    Test the `__call__` method of ThreatDetection with multiple log entries.
    Ensure that:
    - It returns a list of ClassificationResponse objects.
    - All output labels are among the expected threat labels.
    """
    log_entries = [
        "192.168.1.1 - - [03/Apr/2026:10:00:01 +0000] GET /index.html HTTP/1.1 200",
        "10.0.0.2 - - [03/Apr/2026:10:00:02 +0000] POST /login HTTP/1.1 401",
        "172.16.0.5 - - [03/Apr/2026:10:00:03 +0000] GET /?cmd=ls HTTP/1.1 200",
    ]
    out = artifex.threat_detection()(log_entries, device=-1, disable_logging=True)
    assert isinstance(out, list)
    assert all(isinstance(resp, ClassificationResponse) for resp in out)
    assert all(resp.label in THREAT_LABELS for resp in out)
