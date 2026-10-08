"""Partition the existing frozen examples; never select a new training subset."""
import json
import random
import shutil
from pathlib import Path

from scripts import cosmosqa_five_data as original

ROOT, DOMAINS, NAMES = original.ROOT, original.DOMAINS, original.NAMES
CLIENTS = tuple(d + '_' + side for d in DOMAINS for side in ('a', 'b'))
SOURCE = {c: c.rsplit('_', 1)[0] for c in CLIENTS}
MASTER = ROOT / 'outputs/fragmentation_5source_10client_seed42'
CONFIG = ROOT / 'configs/fragmentation_5source_10client_seed42.yaml'
read, rows, sha, dump = original.read, original.rows, original.sha, original.dump
write_rows = original.write_rows
native_key = original.native_key


def config():
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    assert cfg == dict(setting='B', sources=list(DOMAINS),
        source_prepared_dir='dataset/cosmosqa_five_client4000_seed42',
        prepared_dir='dataset/fragmentation_5source_10client_seed42', split_seed=42,
        seed=42, clients_per_source=2, samples_per_client=2000, rounds=10,
        clientlocal_epochs=10, local_epochs_per_round=1, aggregation='standard_ftbase',
        centralized_reference_epoch=6), 'Fragmentation protocol differs'
    return cfg


def partition_indices(size, seed):
    assert size % 2 == 0
    a = sorted(random.Random(seed).sample(range(size), size // 2))
    chosen = set(a)
    return a, [i for i in range(size) if i not in chosen]


def budgets(counts=None, epochs=10, rounds=10):
    counts = counts or {c: 2000 for c in CLIENTS}
    assert tuple(counts) == CLIENTS
    assert all(counts[d + '_a'] == counts[d + '_b'] for d in DOMAINS)
    steps = {c: (n + 7) // 8 for c, n in counts.items()}
    weights = {c: n / sum(counts.values()) for c, n in counts.items()}
    result = dict(client_samples=counts, steps_per_client_epoch=steps,
        client_weights=weights, source_weights={d: weights[d+'_a']+weights[d+'_b'] for d in DOMAINS},
        local_updates=sum(steps.values()) * epochs, fedavg_updates=sum(steps.values()) * rounds,
        local_sample_visits=sum(counts.values()) * epochs,
        fedavg_sample_visits=sum(counts.values()) * rounds)
    if all(n == 2000 for n in counts.values()):
        assert epochs == rounds == 10
        assert all(v == 250 for v in steps.values())
        assert all(v == .1 for v in weights.values())
        assert all(v == .2 for v in result['source_weights'].values())
        assert result['local_updates'] == result['fedavg_updates'] == 25000
        assert result['local_sample_visits'] == result['fedavg_sample_visits'] == 200000
        result['original_5client'] = dict(optimizer_updates=5*500*10, sample_visits=5*4000*10)
    return result


def protected_paths():
    # Protect results and the actual training/evaluation code, including model files.
    paths = list((ROOT / original.config()['prepared_dir']).rglob('*.json*'))
    paths += [original.MASTER / 'final_results.json',
        original.MASTER / 'fedavg/round_diagnostics.json',
        original.MASTER / 'centralized/epoch_06_lora.pt',
        original.MASTER / 'centralized/epoch_10_lora.pt',
        original.MASTER / 'best_saved_analysis/centralized_best_result.json',
        ROOT / 'reports/cosmosqa_five_comparison_seed42.md',
        ROOT / 'reports/cosmosqa_centralized_best_epoch_seed42.md']
    for d in DOMAINS:
        base, local = original.baseline_sources(d)
        paths += [base / 'result.json', base / 'evaluation_provenance.json',
                  local / 'final_lora.pt', local / 'result.json', local / 'evaluation_provenance.json']
    paths += [ROOT / 'models/initial_lora/seed42_r8_alpha32_qv.pt']
    paths += list((ROOT / 'models/models--llama3.2-1B').glob('*'))
    paths += [ROOT / p for p in ('scripts/run_cosmosqa_five_experiments.py',
        'scripts/run_three_dataset_lora.py', 'scripts/cosmosqa_five_data.py',
        'utils/train_utils.py', 'utils/mcq_eval.py', 'utils/mcq_utils.py',
        'utils/model_utils.py', 'utils/seed_utils.py', 'utils/cosmosqa_five_mcq.py',
        'utils/cosmosqa_mcq.py', 'utils/five_client_mcq.py', 'utils/race_mcq.py',
        'alg/ftbase.py', 'alg/base.py', 'configs/cosmosqa_five_client4000_seed42.yaml')]
    return sorted(set(p for p in paths if p.is_file()))


def prepare():
    cfg = config(); folder = ROOT / cfg['prepared_dir']; path = folder / 'manifest.json'
    if path.exists():
        manifest = read(path); verify(manifest); return manifest
    source = ROOT / cfg['source_prepared_dir']
    old = read(source / 'manifest.json')
    for name, digest in old['files'].items(): assert sha(ROOT / name) == digest, name
    assert read(original.MASTER / 'final_results.json')['input_integrity']['passed']
    assert read(original.MASTER / 'best_saved_analysis/centralized_best_result.json')['selected_epoch'] == 6
    manifest = dict(setting='B', split_seed=cfg['split_seed'], files={}, sources={}, clients={},
        source_order=list(DOMAINS), client_order=list(CLIENTS),
        protected_sha256={str(p.relative_to(ROOT)): sha(p) for p in protected_paths()})
    for d in DOMAINS:
        manifest['sources'][d] = {}
        for split in ('train', 'test'):
            src = source / split / (d + '.jsonl'); dst = folder / split / (d + '.jsonl')
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
            manifest['files'][str(dst.relative_to(ROOT))] = sha(dst)
            manifest['sources'][d][split] = dict(path=str(src.relative_to(ROOT)), sha256=sha(src), count=len(rows(src)))
        src = source / 'train' / (d + '.jsonl')
        source_rows, lines = rows(src), src.read_bytes().splitlines(keepends=True)
        assert len(source_rows) == len(lines) == 4000
        for side, indices in zip(('a', 'b'), partition_indices(4000, cfg['split_seed'])):
            c = d + '_' + side; dst = folder / 'clients' / (c + '.jsonl')
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b''.join(lines[i] for i in indices))
            manifest['files'][str(dst.relative_to(ROOT))] = sha(dst)
            manifest['clients'][c] = dict(source=d, client_id=CLIENTS.index(c), size=2000,
                source_train_indices=indices,
                original_samples=[dict(source_train_index=i, native_key=native_key(source_rows[i]),
                    metadata={k: v for k, v in source_rows[i].items()
                        if k in ('id', 'original_id', 'source_file', 'source_file_row', 'source_row_index')}) for i in indices])
    dump(path, manifest); verify(manifest)
    return manifest


def verify(manifest=None):
    cfg = config(); folder = ROOT / cfg['prepared_dir']
    manifest = manifest or read(folder / 'manifest.json')
    assert manifest['split_seed'] == cfg['split_seed']
    for name, digest in {**manifest['protected_sha256'], **manifest['files']}.items():
        assert sha(ROOT / name) == digest, 'Protected or prepared file changed: ' + name
    checks = {}
    for d in DOMAINS:
        original_rows = rows(folder / 'train' / (d + '.jsonl'))
        assert len(original_rows) == len({native_key(r) for r in original_rows}) == 4000
        sets = []; indices_sets = []
        for side, expected in zip(('a', 'b'), partition_indices(4000, cfg['split_seed'])):
            c = d + '_' + side; item = manifest['clients'][c]
            indices = item['source_train_indices']; current = rows(folder / 'clients' / (c + '.jsonl'))
            assert indices == expected and len(current) == item['size'] == 2000
            assert current == [original_rows[i] for i in indices]
            assert [r['native_key'] for r in item['original_samples']] == [native_key(r) for r in current]
            sets.append({native_key(r) for r in current}); indices_sets.append(set(indices))
        assert not sets[0] & sets[1] and not indices_sets[0] & indices_sets[1]
        assert sets[0] | sets[1] == {native_key(r) for r in original_rows}
        assert indices_sets[0] | indices_sets[1] == set(range(4000))
        for split in ('train', 'test'):
            ref = manifest['sources'][d][split]
            assert sha(folder / split / (d + '.jsonl')) == ref['sha256'] == sha(ROOT / ref['path'])
        test = rows(folder / 'test' / (d + '.jsonl'))
        assert not {original.qa_fingerprint(r) for r in original_rows} & {original.qa_fingerprint(r) for r in test}
        checks[d] = dict(original_train=4000, client_a=2000, client_b=2000, intersection=0,
            union=4000, union_equals_original=True, split_seed=cfg['split_seed'], test=len(test), test_unchanged=True)
    audit = dict(passed=True, sources=checks, train_total=20000,
        test_total=sum(v['test'] for v in checks.values()), no_new_or_removed_examples=True,
        shared_test_per_source=True, protected_artifacts_unchanged=True, budget=budgets())
    dump(MASTER / 'data_audit.json', audit)
    return audit
