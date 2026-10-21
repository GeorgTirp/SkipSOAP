"""GPU-port study tests, run by explicit path only (the default suite collects `tests/`).

Serves the copied PARSEC-L tests (test_parsec_copy.py, against gpu_parsec) and the port tests (test_port.py)."""
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
STUDY = Path(__file__).resolve().parents[1]
for path in [STUDY, STUDY/'scripts']:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pytest  # noqa: E402
import torch  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_defaults():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    yield
    torch.set_default_dtype(old)


@pytest.fixture(scope='session')
def mnist():
    from residual_curvature.config import load_config
    from residual_curvature.data import dataset
    c = load_config('configs/mini_vit_allocation.yaml')
    x, y, manifest = dataset(c)
    return c, x, y, manifest
