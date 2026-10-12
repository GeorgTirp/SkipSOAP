"""Stage D harness tests, run by explicit path only."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))


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
    import common
    from residual_curvature.data import dataset
    c = common.study_config()
    x, y, manifest = dataset(c)
    return c, x, y, manifest
