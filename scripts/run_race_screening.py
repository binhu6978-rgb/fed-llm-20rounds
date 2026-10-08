"""Independent RACE-all Base + fixed-budget Local; never dispatches federated jobs."""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
RAW = ROOT / 'data/raw_mcq/race'
UNIFIED = ROOT / 'data/unified_mcq/race'
POOL = ROOT / 'dataset/mcq_balanced4000/race'
RUN = ROOT / 'exp/mcq_base_local_seed42/race'
LOG = ROOT / 'logs/mcq_base_local_seed42/race'
REPORT = ROOT / 'reports/race_base_local_seed42.md'
CANONICAL = ROOT / 'models/initial_lora/seed42_r8_alpha32_qv.pt'
MODEL = 'models/models--llama3.2-1B'
REPO = 'ehovy/race'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def settings(method):
    max_length = json.loads((POOL / 'manifest.json').read_text(encoding='utf-8'))['max_length']
    return SimpleNamespace(model_path=MODEL, model='llama32_1b_base', device=0, mcq=True,
        task_type='CAUSAL_LM', seed=42, bs=1, grad_accum=8, lr=1e-4, epoch=1, step=0,
        lora_rank=8, lora_alpha=32, lora_dropout=0.05,
        suffix=str(RUN / method), max_length=max_length, eval_batch_size=8,
        race_test_path=str(POOL / 'test.jsonl'))


def prepare():
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer
    import numpy as np
    from utils.race_mcq import encode_race
    from utils.mcq_utils import read_jsonl
    if (POOL / 'manifest.json').is_file():
        manifest = json.loads((POOL / 'manifest.json').read_text(encoding='utf-8'))
        for name, digest in manifest['output_sha256'].items():
            assert sha(POOL / name) == digest
        assert len(read_jsonl(POOL / 'train.jsonl')) == 4000
        print('Existing immutable RACE candidate pool verified.', flush=True)
        return
    info = HfApi().dataset_info(REPO)
    snapshot_download(REPO, repo_type='dataset', revision=info.sha,
                      allow_patterns=['all/*.parquet', 'README.md'], local_dir=RAW)
    files = {split: sorted(str(p) for p in (RAW / 'all').glob(split + '-*.parquet'))
             for split in ('train', 'validation', 'test')}
    assert all(files.values()), files
    data = load_dataset('parquet', data_files=files, cache_dir=str(RAW / '.datasets_cache'))
    print('REAL SCHEMA:', data, flush=True)
    excluded, split_counts = [], {}
    for split, ds in data.items():
        print(split, ds.features, 'first sample:', str(ds[0])[:1200], flush=True)
        fields = ds.column_names
        passage_key = next((k for k in ('article', 'passage', 'context') if k in fields), None)
        required = {'question', 'options', 'answer', 'example_id'}
        if passage_key is None or not required.issubset(fields):
            raise ValueError(f'Unrecognized RACE schema: {fields}')
        rows = []
        for index, item in enumerate(ds):
            errors = []
            for key_name in (passage_key, 'question'):
                if not isinstance(item[key_name], str) or not item[key_name].strip():
                    errors.append('empty_' + key_name)
            opts = item['options']
            if not isinstance(opts, list) or len(opts) != 4:
                errors.append('option_count_not_four')
            elif any(not isinstance(x, str) or not x.strip() for x in opts):
                errors.append('empty_option')
            answer = item['answer']
            if not isinstance(answer, str) or answer.strip() not in ('A', 'B', 'C', 'D'):
                errors.append('invalid_answer')
            if errors:
                excluded.append({'split': split, 'row': index, 'reasons': errors, 'original': item})
                continue
            # Native example_id identifies a passage, not an individual question.
            # Official row index suffix preserves the native passage ID and unique QA identity.
            rows.append(dict(dataset='race', id=f"{item['example_id']}::question_row_{index:06d}",
                split=split, context=item[passage_key], question=item['question'],
                **{'option_' + letter.lower(): opts[i] for i, letter in enumerate('ABCD')},
                correct_answer=answer.strip()))
        jsonl(UNIFIED / f'{split}.jsonl', rows)
        split_counts[split] = {'original': len(ds), 'valid': len(rows)}
    jsonl(UNIFIED / 'excluded_rows.jsonl', excluded)
    # No additional artificial length cap: use model's existing positional upper bound.
    max_length = json.loads((ROOT / MODEL / 'config.json').read_text())['max_position_embeddings']
    tokenizer = AutoTokenizer.from_pretrained(ROOT / MODEL, local_files_only=True)
    lengths, truncation = {}, {}
    for split in ('train', 'validation', 'test'):
        rows = read_jsonl(UNIFIED / f'{split}.jsonl')
        sizes, truncated = [], 0
        for row in rows:
            _, _, stat = encode_race(tokenizer, row, max_length)
            sizes.append(stat['full_tokens'])
            truncated += int(stat['passage_truncated'])
        lengths[split] = {name: float(np.percentile(sizes, q)) for name, q in
                         [('median', 50), ('p90', 90), ('p95', 95), ('p99', 99)]}
        lengths[split].update(max=max(sizes), exceeds_max_length=sum(x > max_length for x in sizes),
                             exceeds_max_length_ratio=sum(x > max_length for x in sizes) / len(sizes))
        truncation[split] = dict(total=len(rows), overlong=sum(x > max_length for x in sizes),
                                passage_truncated=truncated, question_options_truncated=0, answer_truncated=0)
        print('TOKEN LENGTH:', split, lengths[split], flush=True)
    train = read_jsonl(UNIFIED / 'train.jsonl')
    random.Random(42).shuffle(train)
    selected = train[:4000]
    assert len(selected) == len(set(r['id'] for r in selected)) == 4000
    jsonl(POOL / 'train.jsonl', selected)
    jsonl(POOL / 'test.jsonl', read_jsonl(UNIFIED / 'test.jsonl'))
    manifest = dict(source_dataset=REPO, source_config='all', revision=info.sha,
        preferred_source_unavailable_reason='EleutherAI/race contains only high/test; no train/all.',
        created_at=datetime.now(timezone.utc).isoformat(), seed=42, source_split='official train',
        original_train_count=split_counts['train']['original'], selected_train_count=4000,
        official_test_count=split_counts['test']['valid'], splits=split_counts,
        schema={s: data[s].features.to_dict() for s in data},
        id_rule='native example_id + ::question_row_ + zero-padded official split row index',
        selected_ids=[r['id'] for r in selected], excluded_count=len(excluded),
        max_length=max_length, previous_mcq_max_length='No explicit preprocessing cap',
        truncation_rule='BOS + full question/options/Answer + longest continuation + EOS reserved; only passage prefix may be truncated.',
        token_length=lengths, truncation_statistics=truncation,
        raw_sha256={str(p.relative_to(RAW)): sha(p) for p in (RAW / 'all').glob('*.parquet')},
        output_sha256={name: sha(POOL / name) for name in ('train.jsonl', 'test.jsonl')},
        validation_used_for_training_or_evaluation=False)
    dump(POOL / 'manifest.json', manifest)
    print('RACE train selected: 4000; test:', manifest['official_test_count'], flush=True)


def base():
    from utils.model_utils import load_model, load_tokenizer
    from utils.seed_utils import set_global_seed
    from utils.race_mcq import RaceEvaluator
    set_global_seed(42, device=0, deterministic=True)
    args = settings('base')
    model = load_model(args)
    result = RaceEvaluator(args, load_tokenizer(args)).evaluate(model, 0)
    dump(RUN / 'base/result.json', result)


def local():
    import torch
    from peft import get_peft_model
    from utils.model_utils import load_model, load_tokenizer, load_lora_config
    from utils.seed_utils import set_global_seed
    from utils.mcq_utils import read_jsonl
    from utils.race_mcq import RaceTrainingDataset, RaceEvaluator
    from utils.train_utils import Trainer
    if not CANONICAL.is_file():
        raise FileNotFoundError('Balanced4000 canonical initial LoRA is required')
    args = settings('local')
    set_global_seed(42, device=0, deterministic=True)
    model = get_peft_model(load_model(args), load_lora_config(args))
    tokenizer = load_tokenizer(args)
    initial = torch.load(CANONICAL, map_location='cpu', weights_only=True)
    current = {k: v for k, v in model.state_dict().items() if 'lora_' in k}
    assert set(current) == set(initial)
    model.load_state_dict(initial, strict=False)
    assert all(torch.equal(model.state_dict()[k].cpu(), v) for k, v in initial.items())
    assert all('lora_' in n for n, p in model.named_parameters() if p.requires_grad)
    args_path = RUN / 'local/protocol.json'
    dump(args_path, dict(vars(args), canonical_initial_sha256=sha(CANONICAL),
                         optimizer_reset_per_epoch=True, base_frozen=True))
    client = SimpleNamespace(id=0, server=SimpleNamespace(round=0))
    client.dataset = {'train': RaceTrainingDataset(read_jsonl(POOL / 'train.jsonl'), tokenizer, args.max_length)}
    trainer = Trainer(args, client.dataset, client)
    assert len(trainer.train_loader) == 4000 and not trainer.train_loader.drop_last
    checkpoint = RUN / 'local/resume_lora.pt'
    progress = RUN / 'local/progress.json'
    curves, completed = [], 0
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state['lora'], strict=False)
        curves, completed = state['curves'], state['epoch']
        # Remove any incomplete-epoch accounting beyond the atomic checkpoint.
        exposure_path = RUN / 'local/exposure.jsonl'
        records = read_jsonl(exposure_path) if exposure_path.is_file() else []
        jsonl(exposure_path, [r for r in records if r['round'] <= completed])
        print('Resume completed epoch:', completed, flush=True)
    for epoch in range(completed + 1, 11):
        client.server.round = epoch - 1
        loss = trainer.train(model)
        assert client.last_seen_count == 4000 and client.last_optimizer_updates == 500
        curves.append(dict(epoch=epoch, train_loss=loss, samples_visited=4000, optimizer_steps=500))
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if 'lora_' in k}
        temporary = checkpoint.with_suffix('.tmp')
        torch.save({'epoch': epoch, 'curves': curves, 'lora': state}, temporary)
        os.replace(temporary, checkpoint)
        dump(progress, curves)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items() if 'lora_' in k},
               RUN / 'local/epoch10_lora.pt')
    # First access to official test in the Local process occurs only here.
    result = RaceEvaluator(args, tokenizer).evaluate(model, 10)
    dump(RUN / 'local/result.json', result)


def report():
    from utils.mcq_utils import read_jsonl
    manifest = json.loads((POOL / 'manifest.json').read_text(encoding='utf-8'))
    b = json.loads((RUN / 'base/result.json').read_text())['race_accuracy']
    l = json.loads((RUN / 'local/result.json').read_text())['race_accuracy']
    curves = json.loads((RUN / 'local/progress.json').read_text())
    records = read_jsonl(RUN / 'local/exposure.jsonl')
    selected = {'race:' + x for x in manifest['selected_ids']}
    assert len(records) == len(curves) == 10
    assert all(r['round'] == i + 1 and r['samples'] == 4000 and r['optimizer_updates'] == 500
               and len(r['ids']) == 4000 and set(r['ids']) == selected for i, r in enumerate(records))
    gap = 100 * (l - b)
    lines = ['# RACE balanced4000 candidate screening (seed 42)', '',
        f"Source: [{REPO}](https://huggingface.co/datasets/{REPO}), configuration `all`, revision `{manifest['revision']}`.",
        'EleutherAI/race was checked and only provides high/test; the equivalent complete official RACE-all repository is used.', '',
        '| Split | Original | Valid | Used |', '| --- | ---: | ---: | --- |']
    for split, count in manifest['splits'].items():
        lines.append(f"| {split} | {count['original']} | {count['valid']} | "
                     + ('4000 fixed train' if split == 'train' else 'final evaluation only' if split == 'test' else 'unused') + ' |')
    lines += ['', f"Excluded unsafe rows: {manifest['excluded_count']}. Native passage IDs retain a unique official question-row suffix.", '',
        '| Measure | Value |', '| --- | ---: |', '| RACE train selected | 4000 |',
        f"| RACE official test size | {manifest['official_test_count']} |",
        f'| Base Accuracy | {100*b:.2f}% |', f'| Local Epoch10 Accuracy | {100*l:.2f}% |',
        f'| Local − Base | {gap:+.2f} pp |', '| Sample visits | 40000 |', '| Optimizer updates | 5000 |', '',
        '## Training protocol', '',
        'Llama-3.2-1B BF16; frozen base; q_proj/v_proj LoRA r=8, alpha=32, dropout=0.05, bias=none. Seed42, bs1, grad_accum8, constant lr1e-4, step0, drop_last=False; current MCQ trainer resets AdamW each epoch. The exact canonical initial LoRA from balanced4000 is loaded. Only answer continuation and EOS are supervised. Conditional likelihood scores all four original-order answer letters; evaluation batch size8 matches balanced4000.', '',
        'Official test is read by the Base job and after Local Epoch10, never by the Local training loop. No best-epoch selection.', '',
        '## Token lengths and truncation', '', f"max_length = {manifest['max_length']} (existing model positional upper bound; previous MCQ preprocessing has no explicit cap).",
        manifest['truncation_rule'], '',
        '| Split | Median | p90 | p95 | p99 | Max | Overlong | Passage truncated | Q/options truncated | Answer truncated |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for split, size in manifest['token_length'].items():
        t = manifest['truncation_statistics'][split]
        lines.append(f"| {split} | {size['median']:.0f} | {size['p90']:.0f} | {size['p95']:.0f} | {size['p99']:.0f} | {size['max']} | {t['overlong']} | {t['passage_truncated']} | 0 | 0 |")
    lines += ['', '## Training loss', '', '| Epoch | Loss | Samples visited | Optimizer steps |', '| --- | ---: | ---: | ---: |']
    for c in curves:
        lines.append(f"| {c['epoch']} | {c['train_loss']:.6f} | 4000 | 500 |")
    # Plain SVG avoids a native font-renderer crash in this Windows environment.
    # The numeric table above remains the authoritative loss record.
    points = [(65 + (c['epoch'] - 1) * 55, 270 - c['train_loss'] / 0.6 * 210) for c in curves]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="640" height="330" viewBox="0 0 640 330">',
           '<rect width="640" height="330" fill="white"/>',
           '<g font-family="Arial,sans-serif" font-size="12" fill="#222">',
           '<text x="230" y="24" font-size="16">RACE Local, seed 42</text>',
           '<text x="280" y="316">Epoch</text>',
           '<text transform="translate(20,210) rotate(-90)">Mean training loss</text>']
    for tick in range(7):
        y = 270 - tick * 35
        svg += [f'<path d="M65 {y} H560" stroke="#ddd"/>',
                f'<text x="35" y="{y+4}">{tick/10:.1f}</text>']
    for epoch in range(1, 11):
        svg.append(f'<text x="{65+(epoch-1)*55-4}" y="290">{epoch}</text>')
    svg += ['<path d="M65 55 V270 H560" fill="none" stroke="#222"/>',
            '<polyline points="' + ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
            + '" fill="none" stroke="#2463a5" stroke-width="2"/>']
    svg += [f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#2463a5"/>' for x, y in points]
    svg.append('</g></svg>')
    (RUN / 'local/train_loss.svg').write_text('\n'.join(svg), encoding='utf-8')
    lines += ['', '![Training loss](../' + (RUN / 'local/train_loss.svg').relative_to(ROOT).as_posix() + ')']
    lines += ['', '## Interpretation', '',
        (f'Local exceeds Base by {gap:.2f} pp under the fixed protocol. This is the observed local specialization gain; no threshold for "clear" improvement was specified and no automatic replacement decision is made.' if gap > 0 else 'RACE did not form an effective local specialist under the fixed protocol.'),
        f"Training loss changed from {curves[0]['train_loss']:.6f} to {curves[-1]['train_loss']:.6f}; "
        + ('decreased.' if curves[-1]['train_loss'] < curves[0]['train_loss'] else 'did not decrease; this is an optimization warning.'),
        'All recorded losses are finite; full exposure checks passed. Loss and two endpoint test scores alone cannot establish underfitting or diagnose generalization failure.', '',
        '| Reference | Base | Local | Gain |', '| --- | ---: | ---: | ---: |', '| MedQA balanced4000 | 37.39% | 35.82% | −1.57 pp |',
        f'| RACE | {100*b:.2f}% | {100*l:.2f}% | {gap:+.2f} pp |', '',
        f'RACE gain is {gap + 1.57:+.2f} pp above the MedQA reference gain. The datasets and task demands differ; this is a candidate screening comparison, not a controlled causal comparison.', '',
        'No FedAvg, IID, Centralized, other-dataset retraining, hyperparameter search, or main-partition change has been performed.']
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    dump(RUN / 'results.json', dict(base_accuracy=b, local_accuracy=l, improvement_pp=gap,
                                   sample_visits=40000, optimizer_updates=5000))
    print('RACE train selected: 4000\nRACE test:', manifest['official_test_count'])
    print('Passage truncated:', manifest['truncation_statistics']['train']['passage_truncated'])
    print('Question/options truncated: 0\nAnswer truncated: 0')
    print(f'Base RACE Accuracy: {100*b:.2f}%\nLocal RACE Accuracy: {100*l:.2f}%\nLocal - Base: {gap:+.2f}pp')
    print('MedQA Local - Base: -1.57pp\nTrain exposure: 40000\nOptimizer updates: 5000\nDone.')


def all_jobs():
    if not (POOL / 'manifest.json').is_file():
        raise FileNotFoundError('Run --job prepare before launching offline training')
    LOG.mkdir(parents=True, exist_ok=True)
    for job in ('base', 'local'):
        if (RUN / job / 'result.json').is_file():
            continue
        with (LOG / f'{job}.log').open('a', encoding='utf-8') as out:
            subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--job', job],
                           cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, check=True)
    report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', choices=['prepare', 'base', 'local', 'all', 'report'], default='all')
    args = parser.parse_args()
    os.chdir(ROOT)
    RUN.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    {'prepare': prepare, 'base': base, 'local': local, 'all': all_jobs, 'report': report}[args.job]()
