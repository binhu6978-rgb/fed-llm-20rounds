"""Seed-42 four-condition paper experiment. No cleaning, tuning or extra baselines."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

DOMAINS = ('ayur', 'legal', 'krishi')
METHODS = ('centralized', 'iid_mixed', 'domain_skewed')
DATA = ROOT / 'dataset/bhashabench_mcq'
RUNS = ROOT / 'exp/bhashabench_llama32'
LOGS = ROOT / 'logs/bhashabench_llama32'
MODEL = ROOT / 'models/models--llama3.2-1B'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def configuration(method, smoke=False):
    return dict(
        alg='centralized' if method == 'centralized' else 'fedit',
        suffix='bhashabench_llama32/' + ('smoke' if smoke else method),
        device=0, dataset=('bhashabench_mcq_smoke' if smoke else 'bhashabench_mcq/' + method),
        model='llama32_1b_base', model_path=str(MODEL), task_type='CAUSAL_LM',
        seed=42, deterministic=True, cn=2 if smoke else (1 if method == 'centralized' else 12),
        sr=1.0, rnd=1 if smoke else 10, test_gap=1, mcq=True,
        mcq_dataset_dir='dataset/bhashabench_mcq_smoke' if smoke else 'dataset/bhashabench_mcq',
        global_test=False, cross_domain_qa=False, round_generation_metrics=False,
        bs=1, grad_accum=8, epoch=1, step=0, lr=1e-4, eval_batch_size=8,
        mode='prototype', lora_rank=8, lora_alpha=32, lora_dropout=0.05,
    )


def prepare():
    import pandas as pd
    import yaml
    from datetime import datetime, timezone
    train, test, sources, unused = {}, {}, {}, {}
    for domain in DOMAINS:
        source = ROOT / 'data/bhashabench_final_pool' / f'{domain}.parquet'
        rows = pd.read_parquet(source).to_dict('records')
        random.Random(42).shuffle(rows)
        train[domain], test[domain] = rows[:2016], rows[2016:2416]
        assert len(train[domain]) == 2016 and len(test[domain]) == 400
        unused[domain] = len(rows) - 2416
        sources[domain] = {'path': str(source.relative_to(ROOT)), 'sha256': sha(source), 'rows': len(rows)}
        write_jsonl(DATA / 'test' / f'{domain}.jsonl', test[domain])
    write_json(DATA / 'train_ids.json', {d: [r['id'] for r in train[d]] for d in DOMAINS})
    pooled = [r for d in DOMAINS for r in train[d]]
    partitions = {'centralized': [pooled], 'iid_mixed': [[] for _ in range(12)], 'domain_skewed': [[] for _ in range(12)]}
    for di, domain in enumerate(DOMAINS):
        shuffled = list(train[domain])
        random.Random(42).shuffle(shuffled)
        for ci in range(12):
            partitions['iid_mixed'][ci].extend(shuffled[ci * 168:(ci + 1) * 168])
        for ci in range(4):
            partitions['domain_skewed'][4 * di + ci].extend(shuffled[ci * 504:(ci + 1) * 504])
    keys = lambda rows: {(r['domain'], r['id']) for r in rows}
    manifest = dict(seed=42, created_at_utc=datetime.now(timezone.utc).isoformat(),
                    source='BhashaBench-V1-derived official evaluation pools',
                    source_files=sources, train_per_domain=2016, test_per_domain=400,
                    dev_created=False, unused_pool_rows=unused,
                    evaluation_domains=list(DOMAINS), conditions={},
                    protocol='Original question and options; original order; letter continuation+EOS; no cleaning or permutation',
                    split_rule='Independent Random(42) shuffle per domain: first 2016 train, next 400 test')
    for method, clients in partitions.items():
        assert keys([r for c in clients for r in c]) == keys(pooled)
        assert sum(map(len, clients)) == len(keys(pooled)) == 6048
        manifest['conditions'][method] = []
        for ci, rows in enumerate(clients):
            random.Random(42 + ci).shuffle(rows)
            assert len(rows) == (6048 if method == 'centralized' else 504)
            counts = {d: sum(r['domain'] == d for r in rows) for d in DOMAINS}
            if method == 'iid_mixed':
                assert set(counts.values()) == {168}
            path = DATA / method / 'train' / f'{ci}.jsonl'
            write_jsonl(path, rows)
            manifest['conditions'][method].append({'id': ci, 'rows': len(rows), 'domain_counts': counts,
                                                     'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
        cfg = ROOT / 'configs' / f'bhashabench_mcq_{method}.yaml'
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(yaml.safe_dump(configuration(method), sort_keys=False), encoding='utf-8')
    manifest['test_files'] = {d: {'ids': [r['id'] for r in test[d]], 'sha256': sha(DATA / 'test' / f'{d}.jsonl')} for d in DOMAINS}
    manifest['model'] = {'path': str(MODEL), 'weights_sha256': sha(MODEL / 'model.safetensors'),
                         'config_sha256': sha(MODEL / 'config.json'), 'tokenizer_sha256': sha(MODEL / 'tokenizer.json')}
    write_json(DATA / 'manifest.json', manifest)
    # One tiny federated smoke run, separate from formal metrics/checkpoints.
    smoke = ROOT / 'dataset/bhashabench_mcq_smoke'
    for ci in range(2):
        write_jsonl(smoke / 'train' / f'{ci}.jsonl', pooled[ci * 8:(ci + 1) * 8])
    for d in DOMAINS:
        write_jsonl(smoke / 'test' / f'{d}.jsonl', test[d][:2])
    (ROOT / 'configs/bhashabench_mcq_smoke.yaml').write_text(yaml.safe_dump(configuration('iid_mixed', True), sort_keys=False), encoding='utf-8')
    print('Prepared fixed 2016 train / 400 test per domain; no dev or cleaning.', flush=True)


def environment():
    import torch
    lines = [f'python={sys.version}', f'executable={sys.executable}',
             f'torch={torch.__version__}', f'cuda_runtime={torch.version.cuda}',
             f'gpu={torch.cuda.get_device_name(0)}', f'bf16_supported={torch.cuda.is_bf16_supported()}']
    distributions = [d for d in importlib.metadata.distributions() if d.metadata.get('Name')]
    for dist in sorted(distributions, key=lambda d: d.metadata['Name'].lower()):
        lines.append(f"{dist.metadata['Name']}=={dist.version}")
    (ROOT / 'reports/runtime_environment.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def base():
    from types import SimpleNamespace
    from utils.seed_utils import set_global_seed
    from utils.model_utils import load_model, load_tokenizer
    from utils.mcq_eval import MCQEvaluator
    args = SimpleNamespace(**configuration('centralized'))
    args.suffix = str(RUNS / 'base')
    set_global_seed(42, device=0, deterministic=True)
    model, tokenizer = load_model(args), load_tokenizer(args)
    result = MCQEvaluator(args, tokenizer).evaluate(model, 0)
    write_json(RUNS / 'base' / 'result.json', result)


def training(method, smoke=False):
    from main import FedSim
    from types import SimpleNamespace
    FedSim(SimpleNamespace(**configuration(method, smoke)))


def records(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def summarize():
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    results = {'base': records(RUNS / 'base/evaluation/metrics.jsonl')[-1]}
    lines = ['# BhashaBench-derived × Llama-3.2-1B：seed42 第一阶段结果', '',
             '## 1. Runtime environment', '', '[完整环境记录](runtime_environment.txt)。使用 fd；torch 未升级。', '',
             '```json', json.dumps(manifest['model'], ensure_ascii=False, indent=2), '```', '',
             '## 2. Data split and client partition', '',
             '| Domain | Train | Test | Unused pool |', '| --- | ---: | ---: | ---: |']
    for d in DOMAINS:
        lines.append(f"| {d} | 2016 | 400 | {manifest['unused_pool_rows'][d]} |")
    lines += ['', '没有 dev。各域独立 Random(42) shuffle，前2016 train、随后400 test；冻结池未修改，未重新清洗/审计，保留原始 question 与 options，包括题干内嵌选项。没有 option permutation。', '',
              'B/C/D 共享6048个train IDs；A/B/C/D共享1200个test IDs。IID每client为168+168+168；Skewed client0–3 Ayur、4–7 Legal、8–11 Krishi，各504。各条件内每题只归一个client。', '',
              '## 3. Training configuration', '',
              'seed42；Llama-3.2-1B base，本地加载，BF16；LoRA q_proj/v_proj，r8/alpha32/dropout0.05/bias none；仅LoRA可训练。常数lr1e-4、AdamW默认betas/eps、weight_decay0.01，每client-call重置optimizer。B每个pooled logical epoch也重置optimizer。', '',
              '10 rounds、epoch1、full participation、microbatch1、accum8；没有step截断，每题每个方法访问10次。C/D每client每轮63 updates，每轮合计756；B每轮756。A/B因子FedAvg公式沿用现有实现。', '',
              'Plain-text Question/A/B/C/D/Answer:；只监督完整空格+字母continuation及EOS；prompt labels=-100。测试计算完整continuation条件logprob，不含EOS，不generate。', '',
              '## 4. Round 1–10 test accuracy curves', '']
    for method in METHODS:
        metrics = records(RUNS / method / 'evaluation/metrics.jsonl')
        losses = records(RUNS / method / 'training_rounds.jsonl')
        assert [r['round'] for r in metrics] == list(range(1, 11))
        exposures = records(RUNS / method / 'exposure.jsonl')
        expected_ids = set()
        for c in manifest['conditions'][method]:
            from utils.mcq_utils import read_jsonl
            expected_ids.update(r['domain'] + ':' + r['id'] for r in read_jsonl(ROOT / c['path']))
        for round_number in range(1, 11):
            ids = [key for row in exposures if row['round'] == round_number for key in row['ids']]
            assert len(ids) == 6048 and set(ids) == expected_ids
            assert sum(r['optimizer_updates'] for r in exposures if r['round'] == round_number) == 756
        results[method] = metrics[-1]
        lines += [f'### {method}', '', '| Round | Train loss | Ayur | Legal | Krishi | Macro | Worst |',
                  '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for m, loss in zip(metrics, losses):
            lines.append(f"| {m['round']} | {loss['train_loss']:.6f} | " + ' | '.join(f"{100*m[k]:.2f}%" for k in ['ayur_accuracy','legal_accuracy','krishi_accuracy','macro_accuracy','worst_domain_accuracy']) + ' |')
        lines.append('')
    lines += ['## 5. Final Round10 results', '', '| Method | Ayur | Legal | Krishi | Macro | Worst |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for method, label in [('base','Base'),('centralized','Centralized-All'),('iid_mixed','FedAvg-IID-Mixed'),('domain_skewed','FedAvg-Domain-Skewed')]:
        lines.append('| ' + label + ' | ' + ' | '.join(f"{100*results[method][k]:.2f}%" for k in ['ayur_accuracy','legal_accuracy','krishi_accuracy','macro_accuracy','worst_domain_accuracy']) + ' |')
    lines += ['', '## 6. Gaps', '', '| Comparison | Gap (percentage points) |', '| --- | ---: |']
    for key in ['ayur_accuracy','legal_accuracy','krishi_accuracy','macro_accuracy']:
        lines.append(f"| IID − Skewed: {key} | {100*(results['iid_mixed'][key]-results['domain_skewed'][key]):+.4f} |")
    central_gap = 100*(results['centralized']['macro_accuracy']-results['domain_skewed']['macro_accuracy'])
    lines += [f'| Centralized − Skewed: macro | {central_gap:+.4f} |', '',
              '## 7. Artifacts and interpretation', '',
              'Configs：configs/bhashabench_mcq_*.yaml；split/partition：dataset/bhashabench_mcq/manifest.json、train_ids.json及各条件train JSONL；logs：logs/bhashabench_llama32/；checkpoints、parameter metadata、exposure与逐题round01–10 predictions/logprobs：exp/bhashabench_llama32/<method>/。Base predictions在base/evaluation/round_00_predictions.csv。', '',
              '每轮读取test是本次用户指定的曲线协议，固定Round10为主结果，没有选best round或据accuracy调整配置/划分。单seed不足以概括优化随机性；这里不额外运行seed或baseline。', '',
              '只说明当前BhashaBench-derived三领域MCQ、Llama-3.2-1B、q/v rank8 LoRA-factor FedAvg、固定协议下客户端领域分布变化与共享模型性能变化的关联。不证明知识遗忘机制、所有FedLLM的普遍问题、因子平均是唯一原因或学到了全新事实。数据来自原官方evaluation，结果不是官方benchmark test成绩。']
    # Provenance check, without performing another data-quality audit.
    for source in manifest['source_files'].values():
        assert sha(ROOT / source['path']) == source['sha256'], 'Frozen source changed'
    initial = [sha(RUNS / method / 'initial_lora.pt') for method in METHODS]
    # torch archives embed filenames/metadata: compare tensor content rather than archive hashes.
    import torch
    states = [torch.load(RUNS / method / 'initial_lora.pt', map_location='cpu', weights_only=True) for method in METHODS]
    assert all(all(torch.equal(states[0][k], state[k]) for k in states[0]) for state in states[1:])
    lines += ['', '所有训练方法初始LoRA张量逐项相同；每轮6048个唯一train IDs全部访问且总updates756；源parquet SHA256未改变。']
    report = ROOT / 'reports/bhashabench_llama32_seed42_results.md'
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(RUNS / 'final_results.json', results)
    print(f"Base macro: {results['base']['macro_accuracy']:.6f}")
    print(f"Centralized Round10 macro: {results['centralized']['macro_accuracy']:.6f}")
    print(f"IID-Mixed Round10 macro: {results['iid_mixed']['macro_accuracy']:.6f}")
    print(f"Domain-Skewed Round10 macro: {results['domain_skewed']['macro_accuracy']:.6f}")
    print(f"IID - Domain-Skewed gap: {results['iid_mixed']['macro_accuracy']-results['domain_skewed']['macro_accuracy']:+.6f}")
    print('Experiment completed.')


def all_runs():
    LOGS.mkdir(parents=True, exist_ok=True)
    environment()
    if not (DATA / 'manifest.json').exists():
        prepare()
    for job in ['smoke', 'base', *METHODS]:
        completed = RUNS / job / ('result.json' if job == 'base' else 'completed.txt')
        if completed.exists():
            continue
        with (LOGS / f'{job}.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--job', job], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        if job == 'smoke':
            import torch
            files = list((RUNS / 'smoke/adapter').rglob('lora_weights.pt'))
            assert files and torch.load(files[0], map_location='cpu', weights_only=True)
    summarize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', choices=['all', 'prepare', 'smoke', 'base', 'summarize', *METHODS], default='all')
    job = parser.parse_args().job
    os.chdir(ROOT)
    if job == 'all':
        all_runs()
    elif job == 'prepare':
        prepare()
    elif job == 'base':
        base()
    elif job == 'smoke':
        training('iid_mixed', True)
    elif job == 'summarize':
        summarize()
    else:
        training(job)
