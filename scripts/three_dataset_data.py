"""Audit official cleaned MCQ sources and prepare full train/combined_test.

Only LogiQA/OpenBookQA/SciQ are accessed. Old raw and unified files are never
changed. Cross-split QA identity ignores A-D permutation to detect reordering.
"""
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ('logiqa', 'openbookqa', 'sciq')
NAMES = dict(zip(DOMAINS, ('LogiQA', 'OpenBookQA', 'SciQ')))
CONFIG = ROOT / 'configs/three_dataset_lora.yaml'
MASTER = ROOT / 'outputs/three_dataset_seed42'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def write_rows(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in value), encoding='utf-8')
    os.replace(temporary, path)


def config():
    import yaml
    value = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    expected = dict(domains=list(DOMAINS), model_path='models/models--llama3.2-1B',
        initial_lora='models/initial_lora/seed42_r8_alpha32_qv.pt', source_dir='data/unified_mcq',
        prepared_dir='dataset/three_dataset_full', seed=42, rounds=10, clientlocal_epochs=10,
        local_epochs_per_round=1, batch_size=1, gradient_accumulation=8,
        learning_rate=1e-4, step=0, drop_last=False, lora_rank=8, lora_alpha=32,
        lora_dropout=0.05, eval_batch_size=8)
    if {k: v for k, v in value.items() if k != 'overlap_policy'} != expected:
        raise ValueError('Training configuration differs from the requested protocol')
    if value['overlap_policy'] not in ('pending', 'remove_train_overlap', 'preserve_all_acknowledged'):
        raise ValueError('Unknown overlap policy')
    return value


def native_key(row):
    return row['dataset'] + ':' + str(row['id'])


def qa_fingerprint(row, ordered=False):
    normalize = lambda text: ' '.join(text.split())
    options = [normalize(row['option_' + letter]) for letter in 'abcd']
    value = [normalize(row['question']), options if ordered else sorted(options),
             options['ABCD'.index(row['correct_answer'])]]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode('utf-8')).hexdigest()


def source_data():
    from scripts import unify_raw_mcq as converter
    cfg = config()
    data, details, source_hashes = {}, {}, {}
    for domain in DOMAINS:
        data[domain], details[domain] = {}, {}
        for split in ('train', 'validation', 'test'):
            path = ROOT / cfg['source_dir'] / domain / (split + '.jsonl')
            current = rows(path)
            assert len(current) == len({str(row['id']) for row in current})
            reconstructed, excluded = [], Counter()
            raw_files = converter.source_files(domain, split)
            if not raw_files:
                raise FileNotFoundError(f'{domain}: official {split} source missing')
            total = 0
            for raw_path, index, raw in converter.raw_records(domain, raw_files):
                total += 1
                try:
                    item_id, question, options, answer = converter.extract(domain, split, index, raw)
                    converter.validate(question, options, answer)
                except (ValueError, TypeError, KeyError, IndexError) as error:
                    excluded[str(error) if isinstance(error, ValueError) else type(error).__name__] += 1
                    continue
                item = dict(dataset=domain, id=item_id, split=split, question=question,
                            **{'option_' + letter: option for letter, option in zip('abcd', options)},
                            correct_answer=answer)
                # Explicitly check SciQ text -> stable insertion -> letter mapping.
                if domain == 'sciq':
                    assert options['ABCD'.index(answer)] == converter.clean(raw['correct_answer'])
                    assert [x for i, x in enumerate(options) if i != 'ABCD'.index(answer)] == [
                        converter.clean(raw[f'distractor{i}']) for i in (1, 2, 3)]
                reconstructed.append(item)
            assert current == reconstructed, (domain, split, 'Existing cleaning differs from raw reconstruction')
            assert all(row['dataset'] == domain and row['split'] == split and row['correct_answer'] in 'ABCD'
                       for row in current)
            data[domain][split] = current
            details[domain][split] = dict(raw_rows=total, cleaned_rows=len(current), exclusions=dict(excluded),
                                         answer_counts=dict(Counter(row['correct_answer'] for row in current)))
            source_hashes[str(path.relative_to(ROOT))] = sha(path)
            for raw_path in raw_files:
                source_hashes[str(raw_path.relative_to(ROOT))] = sha(raw_path)
    return data, details, source_hashes


def audit():
    cfg = config()
    data, cleaning, source_hashes = source_data()
    all_train, all_test = defaultdict(list), defaultdict(list)
    for domain in DOMAINS:
        for row in data[domain]['train']:
            all_train[qa_fingerprint(row)].append(row)
        for split in ('validation', 'test'):
            for row in data[domain][split]:
                all_test[qa_fingerprint(row)].append(row)
    shared = sorted(all_train.keys() & all_test.keys())
    overlaps = [dict(qa_sha256=fingerprint,
                    train_keys=[native_key(row) for row in all_train[fingerprint]],
                    combined_test_keys=[native_key(row) for row in all_test[fingerprint]]) for fingerprint in shared]
    removed = {native_key(row) for fingerprint in shared for row in all_train[fingerprint]} if cfg['overlap_policy'] == 'remove_train_overlap' else set()
    counts, train, test = {}, {}, {}
    for domain in DOMAINS:
        train[domain] = [row for row in data[domain]['train'] if native_key(row) not in removed]
        test[domain] = [dict(row, source_split=row['split'], split='combined_test')
                        for split in ('validation', 'test') for row in data[domain][split]]
        assert len(test[domain]) == len({str(row['id']) for row in test[domain]})
        counts[domain] = dict(original_cleaned_train=len(data[domain]['train']), train=len(train[domain]),
            combined_test=len(test[domain]), official_validation_source=len(data[domain]['validation']),
            official_test_source=len(data[domain]['test']), removed_cross_split_train_rows=sum(
                native_key(row) in removed for row in data[domain]['train']),
            duplicate_train_qa=len(data[domain]['train']) - len({qa_fingerprint(row) for row in data[domain]['train']}),
            optimizer_steps_per_epoch=math.ceil(len(train[domain]) / 8), tail_microbatches=len(train[domain]) % 8)
    train_fps = {qa_fingerprint(row) for domain in DOMAINS for row in train[domain]}
    test_fps = {qa_fingerprint(row) for domain in DOMAINS for row in test[domain]}
    remaining = len(train_fps & test_fps)
    total = sum(count['train'] for count in counts.values())
    weights = {domain: counts[domain]['train'] / total for domain in DOMAINS}
    receipt = read(ROOT / 'docs/migration_files_sha256.json')['files']
    relevant = [key for key in receipt if key.startswith('models/') or key in (
        'utils/mcq_utils.py', 'utils/mcq_eval.py', 'utils/train_utils.py', 'utils/model_utils.py', 'utils/seed_utils.py')]
    for key in relevant:
        assert sha(ROOT / key) == receipt[key], key
    ready = cfg['overlap_policy'] != 'pending' and (remaining == 0 or cfg['overlap_policy'] == 'preserve_all_acknowledged')
    result = dict(protocol_ready=ready, overlap_policy=cfg['overlap_policy'],
        domains=list(DOMAINS), counts=counts, weights=weights, total_train=total,
        combined_test_definition='official validation + official test, no separate validation input',
        raw_reconstruction_exact=True, label_mapping_checked=True, predictions='A/B/C/D',
        logiqa_exclusion_rules_unchanged=True, sciq_correct_and_distractors_verified=True,
        no_train_combined_test_overlap=remaining == 0, original_shared_qa=len(shared),
        remaining_shared_qa=remaining, overlaps=overlaps, removed_train_keys=sorted(removed),
        cleaning=cleaning, source_sha256=source_hashes,
        model_and_existing_mcq_implementation_hashes_verified=True,
        protocol_deviations=([f'{remaining} shared QA identities retained with explicit user-approved preserve-all policy']
                             if remaining and cfg['overlap_policy'] == 'preserve_all_acknowledged' else []))
    dump(MASTER / 'data_audit.json', result)
    lines = ['# Three-dataset pre-training audit', '',
             '| Dataset | Train | Combined Test | Steps/full epoch | FedAvg weight |',
             '| --- | ---: | ---: | ---: | ---: |']
    for domain in DOMAINS:
        c = counts[domain]
        lines.append(f"| {NAMES[domain]} | {c['train']} | {c['combined_test']} | {c['optimizer_steps_per_epoch']} | {weights[domain]:.10f} |")
    lines += ['', f"Overlap policy: `{cfg['overlap_policy']}`. Original cross-split QA identities: {len(shared)}; remaining: {remaining}.",
        'QA identity includes normalized question, option set and correct-answer text; it detects duplicates even when A/B/C/D order changes.',
        'Train-internal repeated content with distinct official row IDs is reported and retained, rather than silently changing full-train exposure.', '',
        'Raw reconstruction exactly matches existing cleaning. OpenBookQA uses main only. SciQ retains the current deterministic correct-answer insertion and distractor order. LogiQA retains the current parsing/exclusion rules.', '',
        'Combined Test contains every cleaned official validation row followed by every cleaned official test row. Source split is retained solely as provenance; the experimental split is combined_test.', '',
        'Existing MCQ Trainer: mcq=True, bs1, epoch1, constant lr1e-4, drop_last=False, step0. AdamW resets each client epoch. Final accumulation windows are divided by their actual microbatch count and call optimizer.step(), so client step counts are ceil(train/8), not equalized.', '',
        'Lifecycle to verify in smoke: save real local LoRA -> own-domain local_after -> sample-count A/B aggregation -> global_after. Every next-round client receives the previous aggregated global. Runtime assertions and event/evaluation provenance receipts audit this ordering.', '',
        f"Formal training ready: {ready}. Training is blocked while the data-policy decision is pending."]
    (ROOT / 'reports/three_dataset_pretraining_audit.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:9]), flush=True)
    print(f"Overlap policy={cfg['overlap_policy']}; shared QA={remaining}; ready={ready}", flush=True)
    return result, train, test


def prepare():
    result, train, test = audit()
    if not result['protocol_ready']:
        raise RuntimeError('Formal data preparation awaits the user-approved overlap policy')
    destination = ROOT / config()['prepared_dir']
    previous = read(destination / 'manifest.json') if (destination / 'manifest.json').exists() else None
    manifest = dict(audit=result, files={})
    for domain in DOMAINS:
        for split, values in (('train', train[domain]), ('test', test[domain])):
            path = destination / split / (domain + '.jsonl')
            if previous:
                assert rows(path) == values, 'Prepared data changed; use a separate experiment directory'
            else:
                write_rows(path, values)
            manifest['files'][str(path.relative_to(ROOT))] = sha(path)
    if previous:
        assert previous == manifest
    else:
        dump(destination / 'manifest.json', manifest)
    return manifest
