"""Reuse three fixed clients; quality-filter and uniformly sample HellaSwag/RACE."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts import three_dataset_balanced4000_data as previous

ROOT = previous.ROOT
DOMAINS = ('logiqa', 'openbookqa', 'sciq', 'hellaswag', 'race')
OLD, NEW = DOMAINS[:3], DOMAINS[3:]
NAMES = dict(zip(DOMAINS, ('LogiQA', 'OpenBookQA', 'SciQ', 'HellaSwag', 'RACE')))
MASTER = ROOT / 'outputs/five_client_balanced4000_seed42'
CONFIG = ROOT / 'configs/five_client_balanced4000_seed42.yaml'
AUDIT_REPORT = ROOT / 'reports/five_client_pretraining_audit.md'
read, rows, sha, dump, write_rows = previous.read, previous.rows, previous.sha, previous.dump, previous.write_rows
native_key = previous.native_key


def config():
    import yaml
    value = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    expected = dict(domains=list(DOMAINS), model_path='models/models--llama3.2-1B',
        initial_lora='models/initial_lora/seed42_r8_alpha32_qv.pt',
        prepared_dir='dataset/five_client_balanced4000_seed42', seed=42,
        train_samples_per_dataset=4000, new_test_samples_per_dataset=1500,
        rounds=10, clientlocal_epochs=10, local_epochs_per_round=1, batch_size=1,
        gradient_accumulation=8, learning_rate=1e-4, step=0, drop_last=False,
        lora_rank=8, lora_alpha=32, lora_dropout=0.05, eval_batch_size=8, participation=1.)
    if value != expected:
        raise ValueError('Five-client configuration differs from the requested protocol')
    return value


def qa_fingerprint(row):
    normalize = lambda x: ' '.join(x.split())
    fields = [normalize(row.get('context', '')), normalize(row['question']),
              sorted(normalize(row['option_'+letter]) for letter in 'abcd')]
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode()).hexdigest()


def preserved_paths():
    # All old Base/Local artifacts plus the immutable old data and reports.
    values = []
    for folder in ('base', 'clientlocal'):
        values += sorted(p for p in (previous.MASTER/folder).rglob('*') if p.is_file())
    values += sorted(p for p in (ROOT/previous.config()['prepared_dir']).rglob('*') if p.is_file())
    values += [previous.CONFIG, ROOT/'reports/three_dataset_balanced4000_seed42.md',
               ROOT/'outputs/three_dataset_balanced4000_seed42/fedavg/result.json',
               ROOT/'scripts/run_three_dataset_lora.py', ROOT/'scripts/run_three_dataset_balanced4000.py',
               ROOT/'scripts/three_dataset_data.py', ROOT/'scripts/three_dataset_balanced4000_data.py',
               ROOT/'utils/train_utils.py', ROOT/'utils/mcq_utils.py', ROOT/'utils/mcq_eval.py',
               ROOT/'utils/race_mcq.py', ROOT/'alg/ftbase.py', ROOT/'utils/seed_utils.py', ROOT/'utils/model_utils.py']
    return sorted(set(values))


def verify_previous():
    assert read(previous.MASTER/'status.json')['stage'] == 'complete'
    old_manifest = read(ROOT/previous.config()['prepared_dir']/'manifest.json')
    for name, digest in old_manifest['files'].items():
        assert sha(ROOT/name) == digest, name
    for folder in ('base', 'clientlocal'):
        protocol = read(previous.MASTER/folder/'protocol.json')
        assert protocol['config'] == previous.config()
        assert protocol['counts'] == {d: 4000 for d in OLD}
        assert protocol['runner_sha256'] == sha(ROOT/'scripts/run_three_dataset_lora.py')
        assert protocol['sequence_runner_sha256'] == sha(ROOT/'scripts/run_three_dataset_balanced4000.py')
        assert protocol['mcq_trainer_sha256'] == sha(ROOT/'utils/train_utils.py')
        assert protocol['aggregation_sha256'] == sha(ROOT/'alg/ftbase.py')
        assert protocol['initial_lora_sha256'] == sha(ROOT/config()['initial_lora'])
        assert (previous.MASTER/folder/'completed.txt').exists()
    result = {str(path.relative_to(ROOT)): sha(path) for path in preserved_paths()}
    manifest_path = ROOT/config()['prepared_dir']/'manifest.json'
    if manifest_path.exists():
        assert result == read(manifest_path)['audit']['preserved_sha256'], 'Old artifacts changed'
    return old_manifest, result


def raw_split(domain, split):
    import pyarrow.parquet as pq
    folder = ROOT/'hellaswag' if domain == 'hellaswag' else ROOT/'data/raw_mcq/race/all'
    files = sorted(folder.glob(split+'*.parquet'))
    assert files, (domain, split, 'raw data unavailable')
    items, schemas, source_hashes = [], {}, {}
    for path in files:
        table = pq.read_table(path)
        schemas[str(path.relative_to(ROOT))] = str(table.schema)
        source_hashes[str(path.relative_to(ROOT))] = sha(path)
        for local_index, raw in enumerate(table.to_pylist()):
            items.append((path, local_index, len(items), raw))
    return items, schemas, source_hashes


def clean_split(domain, split):
    raw, schemas, source_hashes = raw_split(domain, split)
    rejected, groups = [], defaultdict(list)
    for file, local_index, index, item in raw:
        errors = []
        if domain == 'hellaswag':
            question, context, options = item.get('ctx'), '', item.get('endings')
            label = str(item.get('label', ''))
            answer = 'ABCD'[int(label)] if label in ('0','1','2','3') else None
            original_id = item.get('ind')
            row_id = f'{split}:ind:{original_id}::row_{index:06d}'
        else:
            question, context, options = item.get('question'), item.get('article'), item.get('options')
            answer = item.get('answer')
            answer = answer.strip() if isinstance(answer, str) else None
            original_id = item.get('example_id')
            row_id = f'{original_id}::question_row_{index:06d}'
        if not isinstance(question, str) or not question.strip(): errors.append('empty_question_or_context')
        if domain == 'race' and (not isinstance(context, str) or not context.strip()): errors.append('empty_passage')
        if not isinstance(options, list) or len(options) != 4:
            errors.append('choice_count_not_four')
        elif any(not isinstance(x, str) or not x.strip() for x in options):
            errors.append('empty_choice')
        elif len({' '.join(x.split()) for x in options}) != 4:
            errors.append('repeated_choice_text')
        if answer not in ('A','B','C','D'): errors.append('invalid_label')
        if original_id is None or original_id == '': errors.append('missing_native_id')
        provenance = dict(dataset=domain, source_split=split, source_file=str(file.relative_to(ROOT)),
            source_file_row=local_index, source_row_index=index, original_id=original_id, id=row_id)
        if errors:
            rejected.append(dict(provenance, reasons=errors))
            continue
        row = dict(dataset=domain, split=split, id=row_id, question=question,
            **{'option_'+letter: option for letter, option in zip('abcd', options)},
            correct_answer=answer, **{k: v for k, v in provenance.items() if k not in ('dataset','id')})
        if domain == 'race': row['context'] = context
        else: row['source_id'] = item.get('source_id')
        groups[qa_fingerprint(row)].append(row)
    clean = []
    for fingerprint, group in groups.items():
        answers = {' '.join(row['option_'+row['correct_answer'].lower()].split()) for row in group}
        if len(answers) != 1:
            rejected += [dict(id=row['id'], dataset=domain, source_split=split,
                              reasons=['inconsistent_duplicate_labels'], qa_sha256=fingerprint) for row in group]
            continue
        clean.append(group[0])
        rejected += [dict(id=row['id'], dataset=domain, source_split=split,
                          reasons=['normalized_duplicate_qa'], duplicate_of=group[0]['id']) for row in group[1:]]
    assert len(clean) == len({native_key(row) for row in clean})
    details = dict(raw_rows=len(raw), clean_rows=len(clean), rejected_rows=len(rejected),
        rejection_reasons=dict(Counter(reason for row in rejected for reason in row['reasons'])), schemas=schemas)
    return clean, rejected, details, source_hashes


def sequence_statistics(train, test):
    import numpy as np
    from transformers import AutoTokenizer
    from utils import five_client_mcq as encoding
    tokenizer = AutoTokenizer.from_pretrained(ROOT/config()['model_path'], local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    result = {}
    for domain in DOMAINS:
        result[domain] = {}
        for split, current in (('train', train[domain]), ('test', test[domain])):
            lengths, prompts, targets, truncated = [], [], [], 0
            for row in current:
                prefix = encoding.encode_prompt(tokenizer, row)
                target = encoding.continuation_ids(tokenizer, row, row['correct_answer']) + [tokenizer.eos_token_id]
                prompts.append(len(prefix)); targets.append(len(target)); lengths.append(len(prefix)+len(target))
                if domain == 'race': truncated += int(encoding.encoded_race(tokenizer, row)[2]['passage_truncated'])
            result[domain][split] = dict(samples=len(current),
                label_distribution=dict(Counter(row['correct_answer'] for row in current)),
                token_total=sum(lengths), prompt_token_total=sum(prompts), supervised_token_total=sum(targets),
                mean_input_length=float(np.mean(lengths)), median_input_length=float(np.median(lengths)),
                p90_input_length=float(np.percentile(lengths,90)), max_input_length=max(lengths),
                mean_prompt_length=float(np.mean(prompts)), passage_truncated_samples=truncated,
                definition='BOS + original prompt + gold answer continuation + EOS; no padding')
            print('SEQUENCE', domain, split, json.dumps(result[domain][split]), flush=True)
    encoding.race_encoding.cache_clear()
    return result


def audit():
    cfg = config()
    old_manifest, preserved = verify_previous()
    old_folder = ROOT/previous.config()['prepared_dir']
    train = {domain: rows(old_folder/'train'/f'{domain}.jsonl') for domain in OLD}
    test = {domain: rows(old_folder/'test'/f'{domain}.jsonl') for domain in OLD}
    cleaning, source_hashes, selections, excluded = {}, {}, {}, []
    for domain in NEW:
        test_split = 'validation' if domain == 'hellaswag' else 'test'
        clean_train, reject_train, tr_detail, tr_hash = clean_split(domain, 'train')
        clean_test, reject_test, te_detail, te_hash = clean_split(domain, test_split)
        source_hashes.update(tr_hash); source_hashes.update(te_hash)
        excluded += reject_train + reject_test
        selected_test = previous.select_fixed_train(clean_test, seed=42, count=1500)
        test[domain] = [dict(row, split='test') for row in selected_test]
        test_fps = {qa_fingerprint(row) for current in test.values() for row in current}
        eligible = []
        for row in clean_train:
            if qa_fingerprint(row) in test_fps:
                excluded.append(dict(dataset=domain, id=row['id'], source_split='train', reasons=['train_test_duplicate']))
            else: eligible.append(row)
        train[domain] = previous.select_fixed_train(eligible, seed=42, count=4000)
        cleaning[domain] = dict(train=tr_detail, test=te_detail, test_official_split=test_split,
            train_test_removed=len(clean_train)-len(eligible), eligible_train=len(eligible))
        selections[domain] = dict(seed=42, sampling='uniform without replacement, restore official source order',
            train=[dict(id=r['id'], original_id=r['original_id'], source_file=r['source_file'],
                        source_file_row=r['source_file_row'], source_row_index=r['source_row_index']) for r in train[domain]],
            test=[dict(id=r['id'], original_id=r['original_id'], source_file=r['source_file'],
                       source_file_row=r['source_file_row'], source_row_index=r['source_row_index']) for r in test[domain]])
    all_test = {qa_fingerprint(row) for current in test.values() for row in current}
    assert not {qa_fingerprint(row) for current in train.values() for row in current}.intersection(all_test)
    for domain in DOMAINS:
        assert len(train[domain]) == len({native_key(row) for row in train[domain]}) == 4000
        assert len(test[domain]) == len({native_key(row) for row in test[domain]})
        assert all(r['correct_answer'] in 'ABCD' and all(isinstance(r['option_'+l],str) and r['option_'+l].strip()
            for l in 'abcd') for r in train[domain]+test[domain])
    for domain in NEW: assert len(test[domain]) == 1500
    receipt = read(ROOT/'docs/migration_files_sha256.json')['files']
    for name in receipt:
        if name.startswith('models/') or name in ('utils/train_utils.py','utils/mcq_utils.py','utils/mcq_eval.py','utils/race_mcq.py'):
            assert sha(ROOT/name) == receipt[name], name
    counts = {domain: dict(train=len(train[domain]), test=len(test[domain]), optimizer_steps_per_epoch=500,
                          test_definition='official validation sample' if domain=='hellaswag' else
                          'official test sample' if domain=='race' else 'existing full Combined Test') for domain in DOMAINS}
    result = dict(protocol_ready=True, domains=list(DOMAINS), counts=counts, weights={d: .2 for d in DOMAINS},
        seed=42, selections=selections, cleaning=cleaning, preserved_sha256=preserved,
        source_sha256=source_hashes, existing_manifest_sha256=sha(old_folder/'manifest.json'),
        no_train_test_overlap=True, old_base_local_reused=True, quality_filter_only=True,
        race_source='data/raw_mcq/race/all (existing official all config; root race directory absent)',
        race_max_length=read(ROOT/'dataset/mcq_balanced4000/race/manifest.json')['max_length'],
        hellaswag_prompt='Existing MCQ Question: ctx; A/B/C/D endings in original order; raw label 0..3 -> A..D',
        race_prompt='Existing utils.race_mcq passage + question + original choices; no new length cap',
        unused_splits='HellaSwag official test and unselected validation; RACE validation are not used',
        observed_notes=['HellaSwag shared short context strings do not imply same QA; normalized full QA and '
            'source_id are checked separately. RACE example_id is a passage ID; original question row index is retained.'])
    dump(MASTER/'data_audit.json', result)
    write_rows(MASTER/'quality_exclusions.jsonl', excluded)
    lines = ['# Five-client data audit (seed42)', '', '| Client | Train | Test | Definition | Weight |',
        '| --- | ---: | ---: | --- | ---: |']
    for d in DOMAINS:
        lines.append(f"| {NAMES[d]} | 4000 | {len(test[d])} | {counts[d]['test_definition']} | 0.20 |")
    lines += ['', 'Old three clients use byte-identical prepared rows and original Base/Local results. '
        'Old Base/Local/data/source/report SHA256 receipts are verified before and after the new run.', '',
        'New rows use only quality filters and seed42 uniform sampling without replacement. '
        'Original native IDs, raw file/index, selected train/test rows, label distributions and cleaning reasons '
        'are recorded. Train versus test normalized full-QA overlap is zero across all five domains.', '',
        'HellaSwag uses labeled official validation as test; its unlabeled official test is not used. '
        'RACE uses official all/train and all/test, keeps passage and original choices, and preserves existing '
        'model positional bound 131072 (only passage truncation if required). No performance-based selection.', '',
        'Configuration: frozen Llama-3.2-1B BF16, canonical q_proj/v_proj LoRA r8/alpha32/dropout0.05; '
        'lr1e-4, bs1, accumulation8, drop_last=False, step0. Existing AdamW resets per full client epoch. '
        'New HellaSwag/RACE Local10 only; fresh FedAvg10, all 5 clients, sample-count weights 0.2.', '',
        'The MCQ likelihood scorer and aggregation implementation remain unchanged. '
        'Every real pre-aggregation local is evaluated on all five tests; matrices are descriptive transfers, '
        'without causal or mechanism claims.']
    AUDIT_REPORT.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines[:10]), flush=True)
    return result, train, test


def prepare():
    result, train, test = audit()
    destination = ROOT/config()['prepared_dir']
    manifest_path = destination/'manifest.json'
    previous_manifest = read(manifest_path) if manifest_path.exists() else None
    manifest = dict(audit=result, files={})
    for domain in DOMAINS:
        for split, current in (('train',train[domain]),('test',test[domain])):
            path = destination/split/f'{domain}.jsonl'
            if previous_manifest:
                assert rows(path) == current, 'Existing five-client sample changed'
            else: write_rows(path,current)
            manifest['files'][str(path.relative_to(ROOT))] = sha(path)
    if previous_manifest:
        assert previous_manifest['audit'] == result
        stats = read(destination/'sequence_statistics.json')
    else:
        stats = sequence_statistics(train,test)
        dump(destination/'sequence_statistics.json',stats)
        with (destination/'sequence_statistics.csv').open('w',encoding='utf-8',newline='') as stream:
            fields = ['dataset','split','samples','token_total','prompt_token_total','supervised_token_total',
                'mean_input_length','median_input_length','p90_input_length','max_input_length','label_distribution']
            writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
            for domain in DOMAINS:
                for split, value in stats[domain].items():
                    writer.writerow(dict(dataset=domain,split=split,**{k:json.dumps(value[k]) if k=='label_distribution' else value[k]
                        for k in fields if k not in ('dataset','split')}))
    for name in ('sequence_statistics.json','sequence_statistics.csv'):
        manifest['files'][str((destination/name).relative_to(ROOT))] = sha(destination/name)
    if previous_manifest: assert previous_manifest == manifest
    else: dump(manifest_path,manifest)
    return manifest
