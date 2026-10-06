"""All application writes stay inside this repository."""
import datetime
import copy
import json
import os
from pathlib import Path
from .board_config import normalize_board

ROOT = Path(__file__).resolve().parents[2]


def local_path(path):
    p = Path(path).expanduser().resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError(f"Output must remain inside {ROOT}")
    return p


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    p = local_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = local_path(p.with_suffix(p.suffix + '.tmp'))
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, p)


def normalized_identity(identity):
    result = copy.deepcopy(identity)
    if 'board' in result:
        result['board'] = normalize_board(result['board'])
    return result


def open_session(path, identity):
    identity = normalized_identity(identity)
    p = local_path(path)
    manifest = p / 'session.json'
    if manifest.exists():
        old = read_json(manifest)
        if normalized_identity(old['identity']) != identity:
            raise ValueError("Session camera, board, robot settings or capture thresholds changed; create a new session")
    else:
        if p.exists() and any(p.iterdir()):
            raise ValueError("Nonempty directory has no session manifest")
        write_json(manifest, {'created_utc': timestamp(), 'identity': identity})
    return p


def samples(path):
    return sorted(Path(path).glob('sample_*/sample.json'))


def next_sample(path):
    p = local_path(path)
    numbers = [int(x.name.split('_')[1]) for x in p.glob('sample_*') if x.name.split('_')[1].isdigit()]
    out = local_path(p / f'sample_{max(numbers, default=0)+1:03d}')
    out.mkdir(exist_ok=False)
    return out


def record_rejection(path, reason):
    p = local_path(Path(path) / 'rejections.jsonl')
    with p.open('a') as f:
        f.write(json.dumps({'timestamp_utc': timestamp(), 'reason': str(reason)}) + '\n')
