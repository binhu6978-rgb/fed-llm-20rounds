"""Fixed 4000-row samples after approved overlap removal; intact Combined Test."""
import random
from collections import defaultdict
from pathlib import Path

from scripts import three_dataset_data as source

ROOT = source.ROOT
DOMAINS, NAMES = source.DOMAINS, source.NAMES
CONFIG = ROOT / 'configs/three_dataset_balanced4000_seed42.yaml'
MASTER = ROOT / 'outputs/three_dataset_balanced4000_seed42'
AUDIT_REPORT = ROOT / 'reports/three_dataset_balanced4000_pretraining_audit.md'
read, rows, sha, dump, write_rows = source.read, source.rows, source.sha, source.dump, source.write_rows
native_key, qa_fingerprint = source.native_key, source.qa_fingerprint


def config():
    import yaml
    value = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    expected = dict(domains=list(DOMAINS), model_path='models/models--llama3.2-1B',
        initial_lora='models/initial_lora/seed42_r8_alpha32_qv.pt', source_dir='data/unified_mcq',
        prepared_dir='dataset/three_dataset_balanced4000_seed42', seed=42, train_samples_per_dataset=4000,
        sampling='seeded_uniform_without_replacement_restore_source_order', rounds=10,
        clientlocal_epochs=10, local_epochs_per_round=1, batch_size=1, gradient_accumulation=8,
        learning_rate=1e-4, step=0, drop_last=False, lora_rank=8, lora_alpha=32,
        lora_dropout=0.05, eval_batch_size=8, overlap_policy='remove_train_overlap')
    if value != expected:
        raise ValueError('Balanced4000 configuration differs from the authorized protocol')
    return value


def select_fixed_train(candidates, seed=42, count=4000):
    if len(candidates) < count:
        raise ValueError(f'Only {len(candidates)} eligible train rows, need {count}')
    indices = sorted(random.Random(seed).sample(range(len(candidates)), count))
    return [candidates[index] for index in indices]


def audit():
    cfg = config()
    official, cleaning, source_hashes = source.source_data()
    all_train, all_test = defaultdict(list), defaultdict(list)
    for domain in DOMAINS:
        for row in official[domain]['train']:
            all_train[qa_fingerprint(row)].append(row)
        for split in ('validation', 'test'):
            for row in official[domain][split]:
                all_test[qa_fingerprint(row)].append(row)
    shared = all_train.keys() & all_test.keys()
    removed = {native_key(row) for fingerprint in shared for row in all_train[fingerprint]}
    train, test, counts, selections = {}, {}, {}, {}
    for domain in DOMAINS:
        candidates = [row for row in official[domain]['train'] if native_key(row) not in removed]
        train[domain] = select_fixed_train(candidates, seed=cfg['seed'], count=cfg['train_samples_per_dataset'])
        test[domain] = [dict(row, source_split=split, split='combined_test')
                       for split in ('validation', 'test') for row in official[domain][split]]
        assert len(train[domain]) == len({native_key(row) for row in train[domain]}) == 4000
        assert len(test[domain]) == len({native_key(row) for row in test[domain]})
        counts[domain] = dict(original_cleaned_train=len(official[domain]['train']),
            eligible_train_after_overlap_removal=len(candidates), train=4000,
            combined_test=len(test[domain]), official_validation_source=len(official[domain]['validation']),
            official_test_source=len(official[domain]['test']), optimizer_steps_per_epoch=500,
            tail_microbatches=0, removed_cross_split_train_rows=len(official[domain]['train'])-len(candidates))
        selections[domain] = dict(seed=42, eligible_count=len(candidates), selected_count=4000,
            algorithm=cfg['sampling'], selected_ids=[native_key(row) for row in train[domain]])
    train_fps = {qa_fingerprint(row) for domain in DOMAINS for row in train[domain]}
    assert not train_fps.intersection(all_test)
    receipt = read(ROOT / 'docs/migration_files_sha256.json')['files']
    relevant = [key for key in receipt if key.startswith('models/') or key in (
        'utils/mcq_utils.py', 'utils/mcq_eval.py', 'utils/train_utils.py', 'utils/model_utils.py', 'utils/seed_utils.py')]
    for key in relevant:
        assert sha(ROOT / key) == receipt[key], key
    result = dict(protocol_ready=True, domains=list(DOMAINS), counts=counts, total_train=12000,
        weights={domain: 1 / 3 for domain in DOMAINS}, sampling=selections,
        combined_test_definition='all cleaned official validation + official test; unchanged',
        overlap_policy=cfg['overlap_policy'], original_shared_qa=len(shared), remaining_shared_qa=0,
        removed_train_keys=sorted(removed), no_train_combined_test_overlap=True,
        raw_reconstruction_exact=True, label_mapping_checked=True, predictions='A/B/C/D',
        sciq_correct_and_distractors_verified=True, logiqa_exclusion_rules_unchanged=True,
        cleaning=cleaning, source_sha256=source_hashes, model_and_existing_mcq_implementation_hashes_verified=True,
        training_budget_per_method=dict(sample_visits=120000, optimizer_updates=15000),
        protocol_deviations=[])
    dump(MASTER / 'data_audit.json', result)
    lines = ['# Three-dataset fixed4000 pre-training audit', '',
        '| Dataset | Eligible Train | Selected Train | Combined Test | Weight | Steps/epoch |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for domain in DOMAINS:
        c = counts[domain]
        lines.append(f"| {NAMES[domain]} | {c['eligible_train_after_overlap_removal']} | 4000 | "
                     f"{c['combined_test']} | {1/3:.10f} | 500 |")
    lines += ['', 'Before sampling, remove all cross-train/Combined-Test same-QA train rows under the prior '
        'user-approved policy, including the 21 reordered OpenBookQA questions. Then independently sample '
        '4000 eligible rows without replacement with random.Random(42), restoring original source order. '
        'The fixed selected IDs and SHA256 input hashes are immutable and shared by ClientLocal and FedAvg.', '',
        'Combined Test remains the complete cleaned official validation + official test. '
        'No training/Combined-Test QA overlap remains. Raw reconstruction, labels, SciQ ordering and '
        'LogiQA cleaning match the original implementation.', '',
        'Base evaluates the canonical initial LoRA. Independent ClientLocal uses 10 full epochs per '
        'specialist; FedAvg uses 10 rounds × 3 clients × 1 full local epoch. Each client has 500 '
        'updates/epoch, hence 5000 total updates and 40000 visits per method. Each method totals '
        '15000 updates and 120000 visits. Constant lr1e-4, bs1, accumulation8, step0, drop_last=False.', '',
        'Original full-train results and inputs remain in their existing directories. '
        'This fixed4000 run starts fresh from initial LoRA in a separate output root.']
    AUDIT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:8]), flush=True)
    return result, train, test


def prepare():
    result, train, test = audit()
    destination = ROOT / config()['prepared_dir']
    path = destination / 'manifest.json'
    previous = read(path) if path.exists() else None
    manifest = dict(audit=result, files={})
    for domain in DOMAINS:
        for split, values in (('train', train[domain]), ('test', test[domain])):
            current = destination / split / (domain + '.jsonl')
            if previous:
                assert rows(current) == values, 'Fixed sample changed; existing experiment refused'
            else:
                write_rows(current, values)
            manifest['files'][str(current.relative_to(ROOT))] = sha(current)
    if previous:
        assert previous == manifest, 'Data provenance changed; existing experiment refused'
    else:
        dump(path, manifest)
    return manifest
