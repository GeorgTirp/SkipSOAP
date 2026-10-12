import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
import yaml

from .optim.soap import UPSTREAM_COMMIT


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_run(c):
    path = Path(c['output']) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    (path / 'resolved.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        r = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else 'unavailable'
    versions = {p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'scipy', 'matplotlib', 'PyYAML']}
    sources = {str(p.relative_to(root)): checksum(p) for base in ['src', 'third_party', 'configs'] for p in (root / base).rglob('*')
               if p.is_file() and '__pycache__' not in str(p) and p.suffix in ['.py', '.yaml']}
    tree_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    revision = git('rev-parse', 'HEAD')
    write_json(path / 'provenance.json', {'source_revision': revision if revision != 'unavailable' else 'tree-sha256:' + tree_hash,
               'source_tree_sha256': tree_hash, 'git_status': git('status', '--porcelain'),
               'source_checksums': sources, 'versions': versions, 'python': sys.version,
               'platform': platform.platform(), 'machine': platform.machine(), 'processor': platform.processor(),
               'device': c['device'], 'dtype': c['dtype'], 'threads': torch.get_num_threads(),
               'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
               'cudnn_deterministic': torch.backends.cudnn.deterministic,
               'matmul_precision': torch.get_float32_matmul_precision(), 'soap_commit': UPSTREAM_COMMIT,
               'soap_adapter': 'pre-refresh project_back instrumentation; source unmodified',
               'seeds': {k: c[k] for k in ['seed', 'data_seed', 'probe_seed', 'analysis_seed']}})
    return path
