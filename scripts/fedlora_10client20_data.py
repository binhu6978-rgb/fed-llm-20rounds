"""Read the existing fragmentation split for fresh 20-round experiments.

No prepare(), partitioning, writes to data/history, or file hashing. Missing
inputs are fatal; copy the already prepared split from the experiment host.
"""
from pathlib import Path

from scripts import cosmosqa_five_data as original
from scripts.run_four_baselines20_server import read, rows

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ('cosmosqa', 'openbookqa', 'sciq', 'hellaswag', 'race')
CLIENTS = tuple(d + '_' + side for d in DOMAINS for side in ('a', 'b'))
SOURCE = {c: c.rsplit('_', 1)[0] for c in CLIENTS}
COUNTS = dict(zip(DOMAINS, (1500, 1000, 1998, 1500, 1500)))
MASTER = ROOT / 'outputs/fedlora_10client20_seed42'
native_key = original.native_key


def config():
    cfg = dict(original.config())
    cfg.update(source_prepared_dir=cfg['prepared_dir'],
               prepared_dir='dataset/fragmentation_5source_10client_seed42',
               rounds=20, split_seed=42, clients_per_source=2, samples_per_client=2000)
    return cfg


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(
            f'Required existing input missing: {path}. Copy the original frozen '
            'data/model from the experiment host; this runner never creates a split.')


def preflight():
    cfg = config()
    folder, source = ROOT / cfg['prepared_dir'], ROOT / cfg['source_prepared_dir']
    required = [folder / 'manifest.json', source / 'sequence_statistics.json',
        ROOT / cfg['initial_lora'], ROOT / cfg['model_path'] / 'config.json',
        ROOT / 'dataset/mcq_balanced4000/race/manifest.json']
    required += [folder / 'clients' / f'{c}.jsonl' for c in CLIENTS]
    required += [root / split / f'{d}.jsonl' for root in (folder, source)
                 for split in ('train', 'test') for d in DOMAINS]
    for path in required:
        require_file(path)
    model_dir = ROOT / cfg['model_path']
    if not any(model_dir.glob('*.safetensors')) and not any(model_dir.glob('pytorch_model*.bin')):
        raise FileNotFoundError(f'Canonical model weights missing: {model_dir}')
    manifest = read(folder / 'manifest.json')
    assert manifest['setting'] == 'B' and manifest['split_seed'] == 42
    assert manifest['source_order'] == list(DOMAINS) and manifest['client_order'] == list(CLIENTS)
    assert set(manifest['clients']) == set(CLIENTS)
    all_train, all_test = set(), set()
    for d in DOMAINS:
        train = rows(source / 'train' / f'{d}.jsonl')
        test = rows(source / 'test' / f'{d}.jsonl')
        assert train == rows(folder / 'train' / f'{d}.jsonl'), (d, 'source union differs')
        assert test == rows(folder / 'test' / f'{d}.jsonl'), (d, 'full test differs')
        assert len(train) == len({native_key(r) for r in train}) == 4000
        assert len(test) == len({native_key(r) for r in test}) == COUNTS[d]
        assert all(r['dataset'] == d and r['correct_answer'] in 'ABCD' for r in train + test)
        index_sets, id_sets = [], []
        for side in ('a', 'b'):
            c = d + '_' + side
            item = manifest['clients'][c]
            indices = item['source_train_indices']
            current = rows(folder / 'clients' / f'{c}.jsonl')
            assert item['source'] == d and item['client_id'] == CLIENTS.index(c)
            assert len(current) == item['size'] == len(indices) == len(set(indices)) == 2000
            assert all(isinstance(i, int) and 0 <= i < 4000 for i in indices)
            assert current == [train[i] for i in indices], (c, 'stored indices/rows differ')
            assert [r['native_key'] for r in item['original_samples']] == [native_key(r) for r in current]
            index_sets.append(set(indices)); id_sets.append({native_key(r) for r in current})
        assert not index_sets[0] & index_sets[1] and index_sets[0] | index_sets[1] == set(range(4000))
        assert not id_sets[0] & id_sets[1] and id_sets[0] | id_sets[1] == {native_key(r) for r in train}
        all_train.update(original.qa_fingerprint(r) for r in train)
        all_test.update(original.qa_fingerprint(r) for r in test)
    assert not all_train & all_test, 'Normalized train/test QA overlap'
    return cfg


class IndexedDataset:
    """Select the saved indices from the identical full-source encoding."""
    def __init__(self, dataset, indices):
        self.rows = [dataset.rows[i] for i in indices]
        self.encoded = [dataset.encoded[i] for i in indices]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.encoded[index]
