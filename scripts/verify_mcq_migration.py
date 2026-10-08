"""Read-only, portable transfer verification. No ML imports or experiment runs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def path(relative):
    # Existing manifests were produced on Windows; support Linux destinations.
    resolved = (ROOT / relative.replace('\\', '/')).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f'Path is outside workspace: {relative}')
    return resolved


def sha(file):
    h = hashlib.sha256()
    with file.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(file):
    return json.loads(file.read_text(encoding='utf-8-sig'))


def main():
    result = read(path('exp/mcq_base_local_seed42/manifest.json'))
    digests = dict(result['preserved_files_sha256'])
    digests.update(read(path('docs/migration_files_sha256.json'))['files'])
    digests[result['canonical_initial_lora']] = result['canonical_initial_sha256']
    for name, digest in digests.items():
        if sha(path(name)) != digest:
            raise ValueError(f'SHA256 mismatch: {name}')
    for domain in ('race', 'logiqa', 'openbookqa', 'sciq'):
        folder = path('dataset/mcq_balanced4000/' + domain)
        manifest = read(folder / 'manifest.json')
        train = [json.loads(line) for line in (folder / 'train.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        ids = [str(row['id']) for row in train]
        assert len(ids) == len(set(ids)) == 4000, domain
        assert ids == manifest['selected_ids'], domain
        assert all(row['dataset'] == domain and row['split'] == 'train' for row in train), domain
        for name, digest in manifest['output_sha256'].items():
            assert sha(folder / name) == digest, (domain, name)
        assert sha(folder / 'test.jsonl') == sha(path(f'data/unified_mcq/{domain}/test.jsonl')), domain
        print(f'{domain}: frozen 4000 IDs, official test and Base/Local files PASS')
    print(f'Migration file verification PASS ({len(digests)} hash entries).')
    print('No training, evaluation, sampling or client partition was performed.')


if __name__ == '__main__':
    main()
