"""Freeze the exact existing CosmosQA plus four-domain inputs, without resampling."""
import csv
import json
import shutil
from pathlib import Path

from scripts import cosmosqa_data as cosmos
from scripts import five_client_data as previous

ROOT = cosmos.ROOT
DOMAINS = ('cosmosqa', 'openbookqa', 'sciq', 'hellaswag', 'race')
NAMES = dict(zip(DOMAINS, ('CosmosQA', 'OpenBookQA', 'SciQ', 'HellaSwag', 'RACE')))
MASTER = ROOT / 'outputs/cosmosqa_five_client4000_seed42'
CONFIG = ROOT / 'configs/cosmosqa_five_client4000_seed42.yaml'
read, rows, sha, dump, write_rows = cosmos.read, cosmos.rows, cosmos.sha, cosmos.dump, cosmos.write_rows
native_key = previous.native_key
qa_fingerprint = previous.qa_fingerprint


def config():
    import yaml
    result = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    expected = dict(domains=list(DOMAINS), model_path='models/models--llama3.2-1B',
        initial_lora='models/initial_lora/seed42_r8_alpha32_qv.pt',
        prepared_dir='dataset/cosmosqa_five_client4000_seed42', seed=42, train_samples_per_dataset=4000,
        rounds=10, clientlocal_epochs=10, centralized_epochs=10, local_epochs_per_round=1,
        batch_size=1, gradient_accumulation=8, learning_rate=1e-4, step=0, drop_last=False,
        lora_rank=8, lora_alpha=32, lora_dropout=0.05, eval_batch_size=8,
        reuse_all_base_local=True, cross_domain_matrix=False)
    assert result == expected, 'Authorized five-domain protocol differs'
    return result


def source_folder(domain):
    return ROOT / (cosmos.config()['prepared_dir'] if domain == 'cosmosqa' else previous.config()['prepared_dir'])


def prepare():
    cfg = config()
    destination = ROOT / cfg['prepared_dir']
    path = destination / 'manifest.json'
    if path.exists():
        verify(read(path))
        return read(path)
    cosmos.verify()
    old_manifest = read(ROOT / previous.config()['prepared_dir'] / 'manifest.json')
    # Use existing protected hashes; this never invokes preparation/training of old clients.
    for name, digest in old_manifest['files'].items(): assert sha(ROOT / name) == digest, name
    preserved = cosmos.core_hashes()
    source_hashes, files, statistics, selections = {}, {}, {}, {}
    for d in DOMAINS:
        folder = source_folder(d)
        source_hashes[str((folder / 'manifest.json').relative_to(ROOT))] = sha(folder / 'manifest.json')
        source_stats = read(folder / 'sequence_statistics.json')
        statistics[d] = source_stats if d == 'cosmosqa' else source_stats[d]
        selections[d] = {}
        for split in ('train', 'test'):
            source = folder / split / f'{d}.jsonl'
            target = destination / split / f'{d}.jsonl'
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists(): assert sha(target) == sha(source), 'Existing frozen row set changed'
            else: shutil.copyfile(source, target)
            source_hashes[str(source.relative_to(ROOT))] = sha(source)
            files[str(target.relative_to(ROOT))] = sha(target)
            current = rows(target)
            selections[d][split] = [native_key(row) for row in current]
    dump(destination / 'sequence_statistics.json', statistics)
    with (destination / 'sequence_statistics.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = ['dataset', 'split', 'samples', 'token_total', 'mean_input_length',
                  'median_input_length', 'p90_input_length', 'max_input_length', 'label_distribution']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for d, splits in statistics.items():
            for split, value in splits.items():
                writer.writerow(dict(dataset=d, split=split, **{key: json.dumps(value[key]) if key == 'label_distribution'
                    else value[key] for key in fields if key not in ('dataset', 'split')}))
    for name in ('sequence_statistics.json', 'sequence_statistics.csv'):
        files[str((destination / name).relative_to(ROOT))] = sha(destination / name)
    manifest = dict(config=cfg, files=files, source_sha256=source_hashes, preserved_sha256=preserved,
        selected_ids=selections, resampled=False, composition='CosmosQA replaces LogiQA; the other four are byte-identical')
    dump(path, manifest)
    verify(manifest)
    return manifest


def verify(manifest=None):
    cfg = config()
    folder = ROOT / cfg['prepared_dir']
    manifest = manifest or read(folder / 'manifest.json')
    assert manifest['config'] == cfg
    for section in ('files', 'source_sha256', 'preserved_sha256'):
        for name, digest in manifest[section].items(): assert sha(ROOT / name) == digest, name
    train = {d: rows(folder / 'train' / f'{d}.jsonl') for d in DOMAINS}
    test = {d: rows(folder / 'test' / f'{d}.jsonl') for d in DOMAINS}
    expected_test = dict(cosmosqa=1500, openbookqa=1000, sciq=1998, hellaswag=1500, race=1500)
    for d in DOMAINS:
        for split, current, expected in (('train', train[d], 4000), ('test', test[d], expected_test[d])):
            assert len(current) == len({native_key(r) for r in current}) == expected
            assert [native_key(r) for r in current] == manifest['selected_ids'][d][split]
            assert sha(folder / split / f'{d}.jsonl') == sha(source_folder(d) / split / f'{d}.jsonl')
            assert all(r['correct_answer'] in 'ABCD' for r in current)
    train_keys = {qa_fingerprint(row) for current in train.values() for row in current}
    test_keys = {qa_fingerprint(row) for current in test.values() for row in current}
    assert not train_keys & test_keys, 'Cross-domain normalized train/test QA overlap'
    result = dict(passed=True, domains=list(DOMAINS), train_counts={d: len(v) for d, v in train.items()},
        test_counts=expected_test, train_total=20000, test_total=sum(expected_test.values()),
        all_inputs_byte_identical_to_sources=True, all_selected_ids_preserved=True,
        normalized_train_test_qa_overlap=0, resampled=False, old_artifacts_preserved=True)
    dump(MASTER / 'input_integrity.json', result)
    return result


def baseline_sources(domain):
    if domain == 'cosmosqa':
        return cosmos.MASTER / 'base', cosmos.MASTER / 'clientlocal/client_cosmosqa'
    if domain in ('openbookqa', 'sciq'):
        root = previous.previous.MASTER
        return root / 'base', root / 'clientlocal' / ('client_' + domain)
    return previous.MASTER / 'base/new_domains', previous.MASTER / 'clientlocal' / ('client_' + domain)
