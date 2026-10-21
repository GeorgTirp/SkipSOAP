"""Study-wide paths, run creation, source archives, unit markers, light mirrors and device setup.

Isolated study `ot_1000_20261013`. Copied from studies/long_run_20261013/scripts/common.py with paths changed. Imported unchanged and archived with every run: studies/cifar_ggn_20261012/scripts (LIB: cifar, micro, and
the modules its runner's code path uses) and studies/gpu_port_20261012/gpu_parsec. LIB is appended to sys.path and
this study's scripts come first. Scripts write only to this study folder and to $OT1K_DATA.
"""
import copy
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
LAB = STUDY.parents[1]
STUDY_REL = STUDY.relative_to(LAB)
PORT = LAB/'studies'/'gpu_port_20261012'
LIB = LAB/'studies'/'cifar_ggn_20261012'/'scripts'
for _path in [LAB/'scripts', PORT]:
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
if str(LIB) not in sys.path:
    sys.path.append(str(LIB))
# This study's modules first: scripts/ holds another summarize.py, LIB another train.py and common.py.
if sys.path[0] != str(STUDY/'scripts'):
    sys.path.insert(0, str(STUDY/'scripts'))
DATA = Path(os.environ.get('OT1K_DATA', Path.home()/'skipsoap-study-data'/'ot_1000_20261013')).expanduser().resolve()
RUNS = DATA/'runs'
PROTOCOL = STUDY_REL/'PROTOCOL.md'
CONFIG = Path('configs/mini_vit_allocation.yaml')
MIRROR_SUFFIXES = {'.json', '.md', '.png', '.yaml'}

# cuBLAS reads this before its first call; deterministic CUDA matmuls need it (PROTOCOL.md, section 2).
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch  # noqa: E402

from residual_curvature.config import load_config  # noqa: E402
from residual_curvature.provenance import checksum, new_run, write_json  # noqa: E402


def guard():
    """Enforce the isolation rules that a script can check for itself."""
    if not os.path.samefile(Path.cwd(), LAB):
        raise RuntimeError(f'Run from {LAB}')
    if not sys.flags.dont_write_bytecode:
        raise RuntimeError('Set PYTHONDONTWRITEBYTECODE=1 (isolation rule 4)')
    if RUNS.is_relative_to(LAB.resolve()):
        raise RuntimeError('$OT1K_DATA must lie outside the repository')
    RUNS.mkdir(parents=True, exist_ok=True)


def setup_torch():
    """Single CPU thread, deterministic algorithms, no TF32 (PROTOCOL.md, section 2)."""
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('CUBLAS_WORKSPACE_CONFIG must be :4096:8')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')


def environment():
    """Versions and device facts recorded with every run."""
    out = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
           'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
           'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32,
           'float32_matmul_precision': torch.get_float32_matmul_precision(), 'threads': torch.get_num_threads()}
    if torch.cuda.is_available():
        out['gpu'] = torch.cuda.get_device_name(0)
        out['compute_capability'] = list(torch.cuda.get_device_capability(0))
    return out


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize()


def read(path):
    return json.loads(Path(path).read_text())


def write_compact(path, value):
    Path(path).write_text(json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n')


def write_gz(path, value):
    with gzip.open(path, 'wt') as f:
        json.dump(value, f, separators=(',', ':'), allow_nan=False)


def read_gz(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)


def study_config(seed=0, **sections):
    """Study config with output under $OT1K_DATA/runs; nested section updates."""
    c = load_config(CONFIG)
    c['output'] = str(RUNS)
    c['seed'] = seed
    for section, values in sections.items():
        c[section].update(values)
    return c


def data_order_seed(seed):
    """Section 3: s=0 reproduces the existing order (data_seed+1 = 102)."""
    return 102+1000*seed


def ids_sha256(ids):
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


def loaded_sources():
    """Repository scripts and study scripts imported by this process."""
    names = set()
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if not path:
            continue
        path = Path(path).resolve()
        for base in [LAB/'scripts', STUDY/'scripts', LIB, PORT/'gpu_parsec']:
            if path.is_relative_to(base.resolve()):
                names.add(str(path.relative_to(LAB.resolve())))
    return sorted(names)


def create_run(c, protocol, extra_files=()):
    """new_run under $OT1K_DATA/runs, then archive every source with checksums."""
    if Path(c['output']).resolve() != RUNS:
        raise RuntimeError('Run output must be $OT1K_DATA/runs')
    directory = new_run(c)
    sources = read(directory/'provenance.json')['source_checksums']
    for name in [str(PROTOCOL), str(CONFIG), *loaded_sources(), *extra_files]:
        sources[name] = checksum(name)
    for name, digest in sources.items():
        target = directory/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
        if checksum(target) != digest:
            raise RuntimeError(f'Archive copy differs: {name}')
    write_json(directory/'source_archive_checksums.json', sources)
    write_json(directory/'protocol.json', protocol)
    return directory


def verify_archive(directory):
    """Workers attach to a run only if every loaded source is archived unchanged."""
    sources = read(directory/'source_archive_checksums.json')
    for name, digest in sources.items():
        if checksum(name) != digest or checksum(directory/'source'/name) != digest:
            raise RuntimeError(f'Source changed since the run was created: {name}')
    missing = [name for name in loaded_sources() if name not in sources]
    if missing:
        raise RuntimeError(f'Loaded sources missing from the run archive: {missing}')
    return sources


def unit_complete(unit):
    """True only for a completed unit whose recorded file checksums still verify."""
    marker = unit/'complete.json'
    if not marker.exists():
        return False
    for name, digest in read(marker)['checksums'].items():
        if checksum(unit/name) != digest:
            raise RuntimeError(f'Completed unit changed: {unit/name}')
    return True


def prepare_unit(unit):
    """Preserve an interrupted unit directory and start the unit afresh."""
    if unit.exists():
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        unit.rename(unit.with_name(f'{unit.name}.interrupted-{stamp}'))
    unit.mkdir(parents=True)


def finish_unit(unit, summary):
    checksums = {str(p.relative_to(unit)): checksum(p) for p in sorted(unit.rglob('*'))
                 if p.is_file() and p.name != 'complete.json'}
    write_json(unit/'complete.json', {'completed': True, 'finished_utc': datetime.now(timezone.utc).isoformat(),
                                      'summary': summary, 'checksums': checksums})


def mirror(directory):
    """Copy json/md/png/yaml files into studies/.../runs/<run-id>/ with EXTERNAL_PATH.txt."""
    directory = Path(directory)
    target = STUDY/'runs'/directory.name
    for path in sorted(directory.rglob('*')):
        if path.is_file() and path.suffix in MIRROR_SUFFIXES:
            out = target/path.relative_to(directory)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, out)
    target.mkdir(parents=True, exist_ok=True)
    (target/'EXTERNAL_PATH.txt').write_text(str(directory)+'\n')
    return target


def peak_rss_bytes():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.seconds = time.perf_counter()-self.start


def deep(value):
    return copy.deepcopy(value)
