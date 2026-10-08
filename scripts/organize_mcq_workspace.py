"""Preserve the four existing Base/Local baselines before explicit legacy removal.

This script copies and verifies artifacts; it never trains, samples or deletes.
Legacy deletion is performed separately from its explicit, bounded plan.
"""
import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ('race', 'logiqa', 'openbookqa', 'sciq')
OLD_POOL = ROOT / 'dataset/heterogeneous_mcq_4datasets_balanced4000'
OLD_RUN = ROOT / 'exp/heterogeneous_mcq_balanced4000_seed42'
RACE_POOL = ROOT / 'dataset/race_balanced4000_candidate'
RACE_RUN = ROOT / 'exp/race_balanced4000_candidate_seed42'
NEW_POOL = ROOT / 'dataset/mcq_balanced4000'
NEW_RUN = ROOT / 'exp/mcq_base_local_seed42'
NEW_LOG = ROOT / 'logs/mcq_base_local_seed42'
INITIAL = ROOT / 'models/initial_lora/seed42_r8_alpha32_qv.pt'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def copy(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    assert sha(source) == sha(dest), dest


def predictions(source, dest, domain):
    with source.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        selected = [row for row in reader if row['domain'] == domain]
    assert selected
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    return len(selected), sum(int(r['correct']) for r in selected) / len(selected)


def main():
    if (NEW_RUN / 'manifest.json').exists():
        manifest = read(NEW_RUN / 'manifest.json')
        for name, digest in manifest['preserved_files_sha256'].items():
            assert sha(ROOT / name) == digest, name
        assert sha(INITIAL) == manifest['canonical_initial_sha256']
        assert (ROOT / 'reports/workspace_cleanup.json').is_file()
        print('Previously preserved artifacts verified; use existing cleanup plan.')
        return
    original = read(OLD_POOL / 'manifest.json')
    race_manifest = read(RACE_POOL / 'manifest.json')
    copy(OLD_RUN / 'initial_lora.pt', INITIAL)
    initial_hash = sha(INITIAL)
    data_manifest, result_manifest, summary = {}, {}, []
    for domain in DOMAINS:
        index = ('medqa', 'logiqa', 'openbookqa', 'sciq').index(domain) if domain != 'race' else None
        source_train = RACE_POOL / 'train.jsonl' if domain == 'race' else ROOT / original['clients'][index]['path']
        source_test = ROOT / 'data/unified_mcq' / domain / 'test.jsonl'
        domain_pool = NEW_POOL / domain
        copy(source_train, domain_pool / 'train.jsonl')
        copy(source_test, domain_pool / 'test.jsonl')
        train, test = rows(domain_pool / 'train.jsonl'), rows(domain_pool / 'test.jsonl')
        expected_ids = race_manifest['selected_ids'] if domain == 'race' else original['train_sources'][domain]['selected_ids']
        assert len(train) == len(set(str(r['id']) for r in train)) == 4000
        assert [str(r['id']) for r in train] == expected_ids
        assert all(r['dataset'] == domain and r['split'] == 'train' for r in train)
        manifest = dict(race_manifest) if domain == 'race' else dict(
            source_dataset=domain, source_split='official train', seed=42,
            original_train_count=original['train_sources'][domain]['original_train_size'],
            selected_train_count=4000, official_test_count=len(test), selected_ids=expected_ids)
        manifest.update(output_sha256={name: sha(domain_pool / name) for name in ('train.jsonl', 'test.jsonl')},
                        preserved_existing_selection=True, new_sampling_performed=False)
        save(domain_pool / 'manifest.json', manifest)
        data_manifest[domain] = dict(train_count=4000, test_count=len(test),
            manifest=str((domain_pool / 'manifest.json').relative_to(ROOT)),
            manifest_sha256=sha(domain_pool / 'manifest.json'))

        old_local = RACE_RUN / 'local' if domain == 'race' else OLD_RUN / f'local/client_{index:02d}'
        target = NEW_RUN / domain
        base_predictions = RACE_RUN / 'base/evaluation/round_00_predictions.csv' if domain == 'race' else OLD_RUN / 'base/evaluation/round_00_predictions.csv'
        n, base_accuracy = predictions(base_predictions, target / 'base/evaluation/round_00_predictions.csv', domain)
        assert n == len(test)
        source_base = read(RACE_RUN / 'base/result.json' if domain == 'race' else OLD_RUN / 'base/result.json')
        assert abs(base_accuracy - source_base[domain + '_accuracy']) < 1e-12
        base_result = {domain + '_accuracy': base_accuracy, 'macro_accuracy': base_accuracy,
                       'worst_domain_accuracy': base_accuracy}
        save(target / 'base/result.json', base_result)
        save(target / 'base/provenance.json', {'source_result': str(base_predictions.relative_to(ROOT)),
             'source_file_sha256': sha(base_predictions), 'filtered_existing_predictions': True})
        (target / 'base/evaluation/metrics.jsonl').write_text(json.dumps(dict(round=0, **base_result))+'\n', encoding='utf-8')
        local_pred = old_local / 'evaluation/round_10_predictions.csv'
        n, local_accuracy = predictions(local_pred, target / 'local/evaluation/round_10_predictions.csv', domain)
        assert n == len(test)
        source_result = read(old_local / 'result.json')
        assert abs(local_accuracy - source_result.get('own_accuracy', source_result.get(domain+'_accuracy'))) < 1e-12
        if domain != 'race':
            assert source_result['initial_lora_matches_canonical']
        copy(old_local / 'result.json', target / 'local/result.json')
        copy(old_local / 'exposure.jsonl', target / 'local/exposure.jsonl')
        copy(old_local / 'evaluation/metrics.jsonl', target / 'local/evaluation/metrics.jsonl')
        adapter = old_local / ('epoch10_lora.pt' if domain == 'race' else f'client_{index:02d}/lora_weights.pt')
        copy(adapter, target / 'local/epoch10_lora.pt')
        if domain == 'race':
            for name in ('resume_lora.pt', 'progress.json', 'train_loss.svg'):
                copy(old_local / name, target / 'local' / name)
            protocol = read(old_local / 'protocol.json')
            assert protocol['canonical_initial_sha256'] == initial_hash
            protocol.update(suffix=str(target / 'local'), race_test_path=str(domain_pool / 'test.jsonl'))
            save(target / 'local/protocol.json', protocol)
            log_source = ROOT / 'logs/race_balanced4000_candidate_seed42/local.log'
        else:
            copy(old_local / 'parameters.json', target / 'local/parameters.json')
            log_source = ROOT / f'logs/heterogeneous_mcq_balanced4000_seed42/local_{domain}.log'
            losses = re.findall(r'Round (\d+) Client \d+: samples=(\d+) updates=(\d+) loss=([0-9.]+)', log_source.read_text(encoding='utf-8', errors='replace'))
            assert len(losses) == 10
            save(target / 'local/progress.json', [dict(epoch=int(e), samples_visited=int(n), optimizer_steps=int(u), train_loss=float(l)) for e,n,u,l in losses])
        copy(log_source, NEW_LOG / domain / 'local.log')
        exposure = rows(target / 'local/exposure.jsonl')
        keys = {domain + ':' + str(x) for x in expected_ids}
        assert len(exposure) == 10
        assert all(r['round'] == i+1 and r['samples'] == 4000 and r['optimizer_updates'] == 500
                   and len(r['ids']) == 4000 and set(r['ids']) == keys for i,r in enumerate(exposure))
        result_manifest[domain] = dict(base_accuracy=base_accuracy, local_epoch10_accuracy=local_accuracy,
            gain_pp=100*(local_accuracy-base_accuracy), sample_visits=40000, optimizer_updates=5000,
            official_test_count=len(test), original_local_directory=str(old_local.relative_to(ROOT)),
            adapter_sha256=sha(target / 'local/epoch10_lora.pt'),
            selected_data_manifest_sha256=sha(domain_pool / 'manifest.json'))
        summary.append(result_manifest[domain])

    save(NEW_POOL / 'manifest.json', dict(seed=42, datasets=data_manifest, total_train=16000,
         note='Existing frozen selections copied verbatim. No new sampling, split or client partition performed.'))
    preserved_files = {str(p.relative_to(ROOT)): sha(p) for base in (NEW_POOL, NEW_RUN, NEW_LOG)
                       for p in base.rglob('*') if p.is_file()}
    save(NEW_RUN / 'manifest.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        datasets=result_manifest, canonical_initial_lora=str(INITIAL.relative_to(ROOT)),
        canonical_initial_sha256=initial_hash, preserved_files_sha256=preserved_files,
        rerun_required=False, note='Archived existing Base and fixed Epoch10 own-domain Local results; no training/evaluation rerun.'))
    lines = ['# Current four-dataset Base and Local baselines', '',
             'Existing seed42 balanced4000 results are preserved for reuse. No training or evaluation was rerun.', '',
             '| Dataset | Train | Official test | Base | Local Epoch10 | Local − Base |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for domain, r in result_manifest.items():
        lines.append(f"| {domain} | 4000 | {r['official_test_count']} | {100*r['base_accuracy']:.2f}% | {100*r['local_epoch10_accuracy']:.2f}% | {r['gain_pp']:+.2f} pp |")
    base_macro = sum(r['base_accuracy'] for r in summary)/4
    local_macro = sum(r['local_epoch10_accuracy'] for r in summary)/4
    lines += [f'| Macro | 16000 | — | {100*base_macro:.2f}% | {100*local_macro:.2f}% | {100*(local_macro-base_macro):+.2f} pp |', '',
        'Local Macro averages four independent own-dataset specialists. It is not a shared global model.', '',
        '## Conditions for reusing these results', '',
        'Keep the exact selected train IDs and official test files, original option order, prompts and conditional-likelihood scoring, Llama-3.2-1B base, canonical initial LoRA, seed42, r8/alpha32/dropout0.05 q_proj+v_proj, bs1/grad_accum8/lr1e-4, and ten complete epochs. Each Local has 40000 sample visits and 5000 optimizer updates.', '',
        'RACE includes its passage; the other three preserve their existing prompts. This remains a heterogeneous-task/data-source comparison.', '',
        '## Artifact locations', '',
        '- Official original files: `data/raw_mcq/{dataset}/`',
        '- Official unified splits: `data/unified_mcq/{dataset}/`',
        '- Existing fixed4000 selections and full ID manifests: `dataset/mcq_balanced4000/{dataset}/`',
        '- Endpoint predictions, training records and Local adapters: `exp/mcq_base_local_seed42/{dataset}/`',
        '- Local logs: `logs/mcq_base_local_seed42/{dataset}/`',
        '- Canonical initial LoRA: `models/initial_lora/seed42_r8_alpha32_qv.pt`', '',
        'No new four-dataset client partition has been built and no federated run has been started.']
    (ROOT / 'reports/current_mcq_base_local_seed42.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

    targets = []
    for parent, keep in [('dataset', {'mcq_balanced4000'}), ('exp', {'mcq_base_local_seed42'}),
                         ('logs', {'mcq_base_local_seed42'}),
                         ('reports', {'current_mcq_base_local_seed42.md', 'workspace_cleanup.json', 'race_base_local_seed42.md'})]:
        targets += [p for p in (ROOT / parent).iterdir() if p.name not in keep]
    targets += [p for p in (ROOT / 'data').iterdir() if p.name not in {'raw_mcq', 'unified_mcq'}]
    for parent in ('raw_mcq', 'unified_mcq'):
        targets += [p for p in (ROOT / 'data' / parent).iterdir() if p.is_dir() and p.name not in DOMAINS]
    for domain in DOMAINS:
        raw = ROOT / 'data/raw_mcq' / domain
        targets += [p for p in raw.rglob('*') if p.is_dir() and p.name in {'.cache', '.hf_cache', '.datasets_cache', '__pycache__'}]
    for name in ('checkpoints', 'analysis', 'analysis_samples', '__pycache__', 'smoke_4client_run.log'):
        p = ROOT / name
        if p.exists():
            targets.append(p)
    targets += [p for parent in ('scripts', 'utils', 'alg') for p in (ROOT / parent).rglob('__pycache__')]
    normalized = []
    for p in sorted(set(targets), key=lambda x: len(x.parts)):
        p = p.resolve()
        assert p.is_relative_to(ROOT) and p != ROOT
        assert p.parts[len(ROOT.parts)] not in {'models', '.git', 'alg', 'utils'} or p.name == '__pycache__'
        if not any(p.is_relative_to(other) for other in normalized):
            normalized.append(p)
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(), workspace=str(ROOT),
                keep_datasets=list(DOMAINS), preserved_result_manifest=str((NEW_RUN/'manifest.json').relative_to(ROOT)),
                delete_paths=[str(p.relative_to(ROOT)) for p in normalized], deletion_completed=False,
                protected=['models', '.git', 'alg', 'utils', 'scripts', 'configs', 'main.py', 'eval.py'])
    save(ROOT / 'reports/workspace_cleanup.json', plan)
    print('\n'.join(lines[:12]).replace('\u2212', '-'))
    print('Preservation and SHA256 verification: PASS')
    print('Explicit legacy removal targets:', len(normalized))


if __name__ == '__main__':
    main()
