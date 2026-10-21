"""spectral_rotation study tests, run by explicit path only (the default suite collects `tests/`)."""
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
STUDY = Path(__file__).resolve().parents[1]
for path in [STUDY.parent/'gpu_port_20261012', STUDY/'scripts']:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pytest  # noqa: E402
import torch  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_defaults():
    old = torch.get_default_dtype()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    yield
    torch.set_default_dtype(old)
