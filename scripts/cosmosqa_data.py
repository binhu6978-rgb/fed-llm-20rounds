"""Inspect local CosmosQA files, audit quality, and freeze one seed42 selection."""
import csv
import json
import numbers
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts import three_dataset_data as common
from scripts.three_dataset_balanced4000_data import select_fixed_train

ROOT = common.ROOT
DOMAINS, NAMES = ('cosmosqa',), {'cosmosqa': 'CosmosQA'}
CONFIG = ROOT / 'configs/cosmosqa_balanced4000_seed42.yaml'
MASTER = ROOT / 'outputs/cosmosqa_balanced4000_seed42'
read, rows, sha, dump, write_rows = common.read, common.rows, common.sha, common.dump, common.write_rows
SOURCE_OVERRIDE = None


def config():
    import yaml
    value = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    expected = dict(domains=['cosmosqa'], model_path='models/models--llama3.2-1B',
        initial_lora='models/initial_lora/seed42_r8_alpha32_qv.pt', source_dir='data/raw_mcq/cosmosqa',
        prepared_dir='dataset/cosmosqa_balanced4000_seed42', seed=42, train_samples=4000,
        test_samples=1500, clientlocal_epochs=10, batch_size=1, gradient_accumulation=8,
        learning_rate=1e-4, step=0, drop_last=False, lora_rank=8, lora_alpha=32,
        lora_dropout=0.05, eval_batch_size=8, prompt='existing_race_passage_question_choices',
        sampling='seeded_uniform_without_replacement_restore_source_order',
        test_policy='labeled_official_test_else_labeled_validation', scope='base_and_independent_local_only')
    if value != expected:
        raise ValueError('CosmosQA configuration differs from the authorized protocol')
    return value


def fingerprint(row):
    import hashlib
    normal = lambda text: ' '.join(text.split())
    value = [normal(row['context']), normal(row['question']),
             sorted(normal(row['option_' + letter]) for letter in 'abcd')]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def read_file(path):
    if path.suffix == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))
    if path.suffix == '.jsonl':
        return rows(path)
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    raise ValueError(f'Unsupported format: {path}')


def inspect(source_dir):
    source_dir = Path(source_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f'CosmosQA data directory missing: {source_dir}. Pass --source-dir.')
    splits, details = defaultdict(list), []
    for path in sorted(source_dir.rglob('*')):
        if not path.is_file() or path.suffix not in ('.csv', '.jsonl', '.parquet'):
            continue
        match = re.fullmatch(r'(train|test|validation|valid|dev)(?:-\d+-of-\d+)?', path.stem)
        if not match:
            continue
        split = {'valid': 'validation', 'dev': 'validation'}.get(match[1], match[1])
        current = read_file(path)
        if not current:
            raise ValueError(f'Empty official source: {path}')
        columns = sorted(current[0])
        required = {'id', 'context', 'question', 'answer0', 'answer1', 'answer2', 'answer3'}
        if not required.issubset(columns):
            raise ValueError(f'Actual schema differs from cosmos_qa.py: {path}: {columns}')
        offset = len(splits[split])
        splits[split].extend((item, str(path), index, offset + index) for index, item in enumerate(current))
        details.append(dict(file=str(path), format=path.suffix[1:], split=split, rows=len(current),
            fields=columns, valid_accuracy_labels=sum(label(item.get('label')) is not None for item in current),
            sha256=sha(path)))
    if 'train' not in splits:
        raise FileNotFoundError(f'No official train.csv/train parquet/jsonl found in {source_dir}')
    # Refuse accidental duplicate downloads of the same official split.
    for split in splits:
        formats = {record['format'] for record in details if record['split'] == split}
        if len(formats) != 1:
            raise ValueError(f'Multiple formats for {split}; choose a single source directory')
    return splits, details


def label(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value in ('0', '1', '2', '3'):
        return 'ABCD'[int(value)]
    if isinstance(value, numbers.Integral) and 0 <= value <= 3:
        return 'ABCD'[int(value)]
    return None


def clean(raw, split):
    rejected, groups = [], defaultdict(list)
    for item, filename, file_index, index in raw:
        original_id = item.get('id')
        provenance = dict(dataset='cosmosqa', id=f'{split}:{original_id}::row_{index:06d}',
            original_id=original_id, source_split=split, source_file=filename,
            source_file_row=file_index, source_row_index=index)
        reasons = []
        for name in ('context', 'question'):
            if not isinstance(item.get(name), str) or not item[name].strip():
                reasons.append('empty_' + name)
        options = [item.get('answer' + str(i)) for i in range(4)]
        if any(not isinstance(option, str) or not option.strip() for option in options):
            reasons.append('empty_choice')
        elif len({' '.join(option.split()) for option in options}) != 4:
            reasons.append('repeated_choice_text')
        answer = label(item.get('label'))
        if answer is None:
            reasons.append('invalid_label')
        if original_id is None or str(original_id).strip() == '':
            reasons.append('missing_native_id')
        if reasons:
            rejected.append(dict(provenance, reasons=reasons))
            continue
        row = dict(provenance, split=split, context=item['context'], question=item['question'],
            **{'option_' + l: option for l, option in zip('abcd', options)}, correct_answer=answer,
            original_label=int(item['label']))
        groups[fingerprint(row)].append(row)
    selected = []
    for key, group in groups.items():
        gold_texts = {' '.join(row['option_' + row['correct_answer'].lower()].split()) for row in group}
        if len(gold_texts) != 1:
            rejected.extend(dict(row, reasons=['inconsistent_duplicate_labels'], qa_sha256=key) for row in group)
        else:
            selected.append(group[0])
            rejected.extend(dict(row, reasons=['normalized_duplicate_qa'], duplicate_of=group[0]['id'])
                            for row in group[1:])
    # Restore official order after duplicate grouping and reject duplicate native IDs.
    selected.sort(key=lambda row: row['source_row_index'])
    native_groups = defaultdict(list)
    for row in selected:
        native_groups[str(row['original_id'])].append(row)
    conflicting_ids = {key for key, group in native_groups.items() if len(group) > 1}
    kept = []
    for row in selected:
        if str(row['original_id']) in conflicting_ids:
            rejected.append(dict(row, reasons=['conflicting_native_id']))
        else:
            kept.append(row)
    return kept, rejected, dict(raw_rows=len(raw), clean_rows=len(kept), rejected_rows=len(rejected),
        rejection_reasons=dict(Counter(reason for row in rejected for reason in row['reasons'])))


def core_hashes():
    receipt = read(ROOT / 'docs/migration_files_sha256.json')['files']
    protected = {}
    for name, expected in receipt.items():
        if name.startswith('models/') or name in ('utils/train_utils.py', 'utils/mcq_utils.py',
                'utils/mcq_eval.py', 'utils/model_utils.py', 'utils/seed_utils.py', 'utils/race_mcq.py'):
            actual = sha(ROOT / name)
            assert actual == expected, name
            protected[name] = actual
    # Preserve current old results/checkpoints and reports, without running any old experiment.
    for folder in ('outputs/three_dataset_balanced4000_seed42', 'outputs/five_client_balanced4000_seed42'):
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and (path.name in ('result.json', 'final_lora.pt', 'protocol.json', 'manifest.json')
                    or path.as_posix().endswith('fedavg/round_10/global/lora_weights.pt')):
                protected[str(path.relative_to(ROOT))] = sha(path)
    for name in ('reports/three_dataset_balanced4000_seed42.md', 'reports/five_client_fedlora_seed42.md'):
        protected[name] = sha(ROOT / name)
    return protected


def statistics(train, test):
    import numpy as np
    from transformers import AutoTokenizer
    from utils import cosmosqa_mcq as encoding
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config()['model_path'], local_files_only=True)
    result = {}
    for split, current in (('train', train), ('test', test)):
        lengths, prompts, targets, truncations = [], [], [], 0
        for row in current:
            prefix = encoding.encode_prompt(tokenizer, row)
            target = encoding.continuation_ids(tokenizer, row, row['correct_answer']) + [tokenizer.eos_token_id]
            metadata = encoding.encoded(tokenizer, json.dumps(row, ensure_ascii=False, sort_keys=True))[2]
            prompts.append(len(prefix)); targets.append(len(target)); lengths.append(len(prefix) + len(target))
            truncations += int(metadata['passage_truncated'])
            assert not metadata['question_options_truncated'] and not metadata['answer_truncated']
        result[split] = dict(samples=len(current), label_distribution={l: sum(r['correct_answer'] == l for r in current) for l in 'ABCD'},
            token_total=sum(lengths), prompt_token_total=sum(prompts), supervised_token_total=sum(targets),
            mean_input_length=float(np.mean(lengths)), median_input_length=float(np.median(lengths)),
            p90_input_length=float(np.percentile(lengths, 90)), max_input_length=max(lengths),
            passage_truncated_samples=truncations, definition='BOS + existing passage prompt + gold continuation + EOS; no padding')
        print('SEQUENCE', split, json.dumps(result[split]), flush=True)
    encoding.encoded.cache_clear()
    return result


def prepare():
    cfg = config()
    destination = ROOT / cfg['prepared_dir']
    manifest_path = destination / 'manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        verify(manifest)
        if SOURCE_OVERRIDE and str(Path(SOURCE_OVERRIDE).resolve()) != manifest['audit']['source_dir']:
            raise ValueError('Frozen source directory differs; refusing resampling')
        return manifest
    source_dir = Path(SOURCE_OVERRIDE or ROOT / cfg['source_dir']).resolve()
    raw, inspection = inspect(source_dir)
    official_test = raw.get('test', [])
    test_has_labels = bool(official_test) and all(label(item.get('label')) is not None for item, *_ in official_test)
    test_split = 'test' if test_has_labels else 'validation'
    if test_split not in raw:
        raise ValueError('Official test lacks accuracy labels and no labeled validation/valid source exists')
    train_clean, train_rejected, train_quality = clean(raw['train'], 'train')
    test_clean, test_rejected, test_quality = clean(raw[test_split], test_split)
    test = [dict(row, split='test') for row in select_fixed_train(test_clean, count=1500)]
    test_keys = {fingerprint(row) for row in test}
    test_native_ids = {str(row['original_id']) for row in test}
    # Preserve all selected test rows; remove matching train before sampling.
    eligible, overlap_rejected = [], []
    for row in train_clean:
        reasons = []
        if fingerprint(row) in test_keys: reasons.append('train_test_duplicate')
        if str(row['original_id']) in test_native_ids: reasons.append('train_test_native_id_overlap')
        if reasons: overlap_rejected.append(dict(row, reasons=reasons))
        else: eligible.append(row)
    train = select_fixed_train(eligible, count=4000)
    assert not {fingerprint(row) for row in train} & test_keys
    protected = core_hashes()
    stats = statistics(train, test)
    audit = dict(source_dir=str(source_dir), inspected_files=inspection, test_official_split=test_split,
        official_test_usable=test_has_labels, unused='Unused validation rows; unlabeled official test is never evaluated',
        train_source='official train', train_count=4000, test_count=1500, seed=42,
        sampling=cfg['sampling'], quality=dict(train=train_quality, test=test_quality),
        train_overlap_removed=len(overlap_rejected), eligible_train=len(eligible),
        full_clean_qa_overlap=len({fingerprint(r) for r in train_clean} & {fingerprint(r) for r in test_clean}),
        selected_qa_overlap=0, selected_native_id_overlap=0,
        selected_shared_contexts=len({' '.join(r['context'].split()) for r in train} & {' '.join(r['context'].split()) for r in test}),
        protected_sha256=protected, loader_sha256=sha(ROOT / 'cosmos_qa.py'), quality_only=True,
        prompt='Unchanged utils.race_mcq.race_prompt: Passage + Question + A/B/C/D + Answer',
        evaluator='Unchanged utils.mcq_eval.MCQEvaluator.evaluate')
    selected = {split: [dict(id=r['id'], original_id=r['original_id'], source_split=r['source_split'],
        source_file=r['source_file'], source_file_row=r['source_file_row'], source_row_index=r['source_row_index'])
        for r in current] for split, current in (('train', train), ('test', test))}
    write_rows(MASTER / 'quality_exclusions.jsonl', train_rejected + test_rejected + overlap_rejected)
    manifest = dict(config=cfg, audit=audit, selected=selected, files={})
    for split, current in (('train', train), ('test', test)):
        path = destination / split / 'cosmosqa.jsonl'
        write_rows(path, current)
        manifest['files'][str(path.relative_to(ROOT))] = sha(path)
    dump(destination / 'sequence_statistics.json', stats)
    manifest['files'][str((destination / 'sequence_statistics.json').relative_to(ROOT))] = sha(destination / 'sequence_statistics.json')
    dump(manifest_path, manifest)
    dump(MASTER / 'data_audit.json', audit)
    verify(manifest)
    return manifest


def verify(manifest=None):
    cfg = config()
    folder = ROOT / cfg['prepared_dir']
    manifest = manifest or read(folder / 'manifest.json')
    assert manifest['config'] == cfg
    for path, expected in manifest['files'].items():
        assert sha(ROOT / path) == expected, path
    for path, expected in manifest['audit']['protected_sha256'].items():
        assert sha(ROOT / path) == expected, path
    assert sha(ROOT / 'cosmos_qa.py') == manifest['audit']['loader_sha256']
    for item in manifest['audit']['inspected_files']:
        assert sha(item['file']) == item['sha256'], item['file']
    train, test = rows(folder / 'train/cosmosqa.jsonl'), rows(folder / 'test/cosmosqa.jsonl')
    for split, current, count in (('train', train, 4000), ('test', test, 1500)):
        assert len(current) == len({r['id'] for r in current}) == len({str(r['original_id']) for r in current}) == count
        assert len({fingerprint(r) for r in current}) == count
        for row in current:
            assert row['correct_answer'] == 'ABCD'[row['original_label']]
            assert row['context'].strip() and row['question'].strip()
        assert [r['id'] for r in current] == [r['id'] for r in manifest['selected'][split]]
    assert not {fingerprint(r) for r in train} & {fingerprint(r) for r in test}
    assert not {str(r['original_id']) for r in train} & {str(r['original_id']) for r in test}
    # Reconstruct every selected row from its native file/index, without trusting copied fields.
    source_rows = {item['file']: read_file(Path(item['file'])) for item in manifest['audit']['inspected_files']}
    for row in train + test:
        original = source_rows[row['source_file']][row['source_file_row']]
        assert str(row['original_id']) == str(original['id'])
        assert row['context'] == original['context'] and row['question'] == original['question']
        assert row['correct_answer'] == label(original['label'])
        assert all(row['option_' + l] == original['answer' + str(i)] for i, l in enumerate('abcd'))
    result = dict(passed=True, train=4000, test=1500, qa_overlap=0, native_id_overlap=0,
        selected_raw_reconstruction_exact=True, existing_core_and_results_preserved=True)
    dump(MASTER / 'input_integrity.json', result)
    return result
