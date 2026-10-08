"""Download every repository file at an immutable Hugging Face revision."""
import os
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--revision', default=None)
args = parser.parse_args()
repo = 'virtuoussy/Multi-subject-RLVR'
raw = ROOT / 'data/raw/multi_subject_rlvr'
out = ROOT / 'analysis/multi_subject_rlvr'
out.mkdir(parents=True, exist_ok=True)
existing = out / 'download_metadata.json'
revision = args.revision
if revision is None and existing.exists():
    revision = json.loads(existing.read_text(encoding='utf-8'))['revision']
api = HfApi()
info = api.dataset_info(repo, revision=revision, files_metadata=True)
print('Pinned revision:', info.sha, flush=True)
snapshot_download(repo_id=repo, repo_type='dataset', revision=info.sha,
                  local_dir=str(raw), max_workers=2)
files = []
for sibling in info.siblings:
    path = raw / sibling.rfilename
    if not path.is_file():
        raise RuntimeError(f'Missing repository file: {path}')
    if sibling.size is not None and path.stat().st_size != sibling.size:
        raise RuntimeError(f'Size mismatch: {path}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lfs = sibling.lfs
    expected = getattr(lfs, 'sha256', None) if lfs else None
    if expected and digest != expected:
        raise RuntimeError(f'LFS SHA256 mismatch: {path}')
    files.append({'path': sibling.rfilename, 'bytes': path.stat().st_size,
                  'sha256': digest, 'lfs_sha256': expected})
metadata = {'repo_id': repo, 'revision': info.sha,
            'download_date_utc': datetime.now(timezone.utc).isoformat(),
            'license': info.card_data.to_dict().get('license') if info.card_data else None,
            'card_data': info.card_data.to_dict() if info.card_data else {},
            'complete_repository_download': True, 'files': files}
existing.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
