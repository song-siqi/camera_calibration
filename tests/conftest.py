from pathlib import Path
import tempfile
import pytest
from camera_calibration.storage import ROOT


@pytest.fixture
def local_tmp():
    cache = ROOT / '.cache' / 'tests'
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as path:
        yield Path(path)
