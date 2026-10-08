"""Sixteen independent ClientLocal LoRA baselines on the frozen dataset-skewed partition.

Uses the existing fd environment, MCQ trainer/evaluator and exact FedAvg initial
LoRA. Formal clients read only their own official test after epoch 10.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from scripts.run_heterogeneous_mcq_4datasets import (
    DATA, DOMAINS, ROOT as PROJECT_ROOT, RUNS, configuration, key, read_jsonl,
)

assert ROOT == PROJECT_ROOT
INITIAL = RUNS / 'dataset_skewed/initial_lora.pt'
OUT = ROOT / 'checkpoints/clientlocal_4datasets_seed42'
SMOKE = ROOT / 'checkpoints/clientlocal_4datasets_seed42_smoke'
LOGS = ROOT / 'logs/clientlocal_4datasets_seed42'
REPORT = ROOT / 'reports/clientlocal_4datasets_seed42_results.md'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def manifest():
    data = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    clients = data['conditions']['dataset_skewed']
    assert len(clients) == 16 and INITIAL.is_file()
    for ci, entry in enumerate(clients):
        domain = DOMAINS[ci // 4]
        path = ROOT / entry['path']
        assert entry['client'] == ci and sha(path) == entry['sha256']
        assert entry['dataset_counts'][domain] == entry['size']
        assert all(entry['dataset_counts'][other] == 0 for other in DOMAINS if other != domain)
    return data


def run_client(ci, smoke=False):
    import torch
    from alg.clientlocal import Client, Server
    from utils.mcq_eval import MCQEvaluator
    from utils.seed_utils import set_global_seed

    data = manifest()
    domain = DOMAINS[ci // 4]
    entry = data['conditions']['dataset_skewed'][ci]
    output = (SMOKE if smoke else OUT) / f'client_{ci:02d}'
    output.mkdir(parents=True, exist_ok=True)
    config = configuration('dataset_skewed')
    config.update(alg='clientlocal', suffix=str(output), cn=1,
                  mcq_domains=[domain], final_only_test=True, skip_mcq_test=False)
    args = SimpleNamespace(**config)
    set_global_seed(42, device=0, deterministic=True)
    client = Client(ci, args)
    full_size = len(client.dataset['train'])
    assert full_size == entry['size']
    if smoke:
        dataset = client.dataset['train']
        dataset.rows = dataset.rows[:8]
        dataset.encoded = dataset.encoded[:8]
    set_global_seed(42, device=0, deterministic=True)
    server = Server(args, [client], INITIAL)
    assert server.global_evaluator is None, 'Official test must remain unread during training'
    assert all('lora_' in name for name, param in server.model.named_parameters() if param.requires_grad)
    epochs = 1 if smoke else 10
    for epoch in range(1, epochs + 1):
        loss = server.train_client_epoch(client, epoch)
        assert client.last_seen_count == (8 if smoke else full_size)
        assert client.last_optimizer_updates == math.ceil(client.last_seen_count / config['grad_accum'])
        print(f'Client {ci:02d} epoch {epoch}/{epochs}: samples={client.last_seen_count} '
              f'updates={client.last_optimizer_updates} loss={loss:.6f}', flush=True)
    checkpoint = output / 'lora_weights.pt'
    torch.save(server.client_loras[ci], checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert set(saved) == set(server.initial_lora)
    assert all(torch.equal(saved[name], server.client_loras[ci][name]) for name in saved)
    # The official test is first read here, after all requested local epochs.
    evaluator = MCQEvaluator(args, client.tokenizer)
    assert evaluator.domains == (domain,)
    if smoke:
        evaluator.rows[domain] = evaluator.rows[domain][:2]
        evaluator.encoded[domain] = evaluator.encoded[domain][:2]
    metrics = evaluator.evaluate(server.model, epochs)
    result = {'client': ci, 'dataset': domain, 'train_size': full_size,
              'trained_epochs': epochs, 'initial_lora_sha256': sha(INITIAL),
              'partition_sha256': entry['sha256'],
              'checkpoint': str(checkpoint.relative_to(ROOT)),
              'own_dataset_accuracy': metrics[domain + '_accuracy'],
              'official_test_count': len(evaluator.rows[domain]), 'metrics': metrics}
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'completed.txt').write_text(f'Client {ci}: {epochs} complete epochs and one own-dataset test.\n', encoding='utf-8')
    print(f'Client {ci:02d} {domain} own-dataset accuracy: {100*result["own_dataset_accuracy"]:.2f}%', flush=True)


def summarize():
    data = manifest()
    initial_hash = sha(INITIAL)
    results = []
    for ci in range(16):
        output = OUT / f'client_{ci:02d}'
        assert (output / 'completed.txt').is_file() and (output / 'lora_weights.pt').is_file()
        result = json.loads((output / 'result.json').read_text(encoding='utf-8'))
        entry = data['conditions']['dataset_skewed'][ci]
        assert result['client'] == ci and result['dataset'] == DOMAINS[ci // 4]
        assert result['train_size'] == entry['size'] and result['trained_epochs'] == 10
        assert result['partition_sha256'] == entry['sha256'] and result['initial_lora_sha256'] == initial_hash
        expected_ids = {key(row) for row in read_jsonl(ROOT / entry['path'])}
        exposure = read_jsonl(output / 'exposure.jsonl')
        assert len(exposure) == 10 and [row['round'] for row in exposure] == list(range(1, 11))
        for row in exposure:
            assert row['client'] == ci and row['samples'] == entry['size']
            assert row['optimizer_updates'] == math.ceil(entry['size'] / 8)
            assert len(row['ids']) == len(set(row['ids'])) == entry['size'] and set(row['ids']) == expected_ids
        eval_rows = read_jsonl(output / 'evaluation/metrics.jsonl')
        assert len(eval_rows) == 1 and eval_rows[0]['round'] == 10
        assert result['own_dataset_accuracy'] == eval_rows[0][result['dataset'] + '_accuracy']
        results.append(result)
    means = {domain: sum(row['own_dataset_accuracy'] for row in results if row['dataset'] == domain) / 4
             for domain in DOMAINS}
    macro = sum(means.values()) / 4
    fed = json.loads((RUNS / 'final_results.json').read_text(encoding='utf-8'))['dataset_skewed']
    lines = [
        '# Four-dataset ClientLocal × Llama-3.2-1B (seed 42)', '',
        '## 1. Purpose and data', '',
        'This supplement measures each independent client’s accuracy on its **own dataset** after 10 complete local epochs. It reuses the frozen 16-client Dataset-Skewed partition and official test files without new cleaning or splits. Clients 0–3: MedQA; 4–7: LogiQA; 8–11: OpenBookQA; 12–15: SciQ. The four clients of a dataset remain separate.', '',
        f'All 16 runs start from the same local Llama-3.2-1B base and the exact Dataset-Skewed FedAvg initial LoRA (`{initial_hash}`). Each run is a separate process, loads only its own train file, and never aggregates or inherits another client’s adapter.', '',
        '## 2. Configuration and evaluation', '',
        'BF16; q_proj/v_proj LoRA r=8, alpha=32, dropout=0.05, bias=none; frozen base; seed=42; constant lr=1e-4; AdamW; batch size 1; gradient accumulation 8. Each local epoch visits every local sample once with `drop_last=False`; the existing trainer resets AdamW each epoch, matching the FedAvg per-round local optimizer reset. Each training example is seen 10 times. The MCQ prompt, space-plus-letter continuation + EOS target and conditional log-likelihood A/B/C/D evaluator are unchanged. Each client reads only its own official test after epoch 10 and evaluates once; no best checkpoint selection or cross-dataset matrix.', '',
        '## 3. Each client: own-dataset test', '',
        '| Client | Dataset | Train size | Own-dataset test accuracy |',
        '| ---: | --- | ---: | ---: |',
    ]
    for row in results:
        lines.append(f"| {row['client']} | {row['dataset']} | {row['train_size']:,} | {100*row['own_dataset_accuracy']:.2f}% |")
    lines += ['', '## 4. Four-client averages and ClientLocal Macro', '',
              '| Dataset | Clients | Local average |', '| --- | --- | ---: |']
    for di, domain in enumerate(DOMAINS):
        lines.append(f'| {domain} | {di*4}–{di*4+3} | {100*means[domain]:.2f}% |')
    lines += ['', f'**ClientLocal Macro: {100*macro:.2f}%** (unweighted mean of the four local dataset averages).', '',
              '## 5. Comparison with Dataset-Skewed FedAvg Round 10', '',
              '| Method | MedQA | LogiQA | OpenBookQA | SciQ | Macro |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    lines.append('| ClientLocal | ' + ' | '.join(f'{100*means[d]:.2f}%' for d in DOMAINS)
                 + f' | {100*macro:.2f}% |')
    lines.append('| FedAvg-Dataset-Skewed | ' + ' | '.join(f'{100*fed[d+"_accuracy"]:.2f}%' for d in DOMAINS)
                 + f' | {100*fed["macro_accuracy"]:.2f}% |')
    lines += ['', '| ClientLocal − FedAvg-Skewed | Difference (percentage points) |',
              '| --- | ---: |']
    for domain in DOMAINS:
        lines.append(f'| {domain} | {100*(means[domain]-fed[domain+"_accuracy"]):+.4f} |')
    lines.append(f'| Macro | {100*(macro-fed["macro_accuracy"]):+.4f} |')
    lines += ['', '## 6. Interpretation boundary', '',
              'ClientLocal own-dataset averages combine four **different local models** per dataset, whereas FedAvg scores one shared global LoRA on each dataset. If a local average is higher, this indicates that some independently learned own-dataset capability is not fully retained by this shared model under the current budget; it does not by itself establish a parameter conflict mechanism or forgetting. If a local average is lower, the single client had only about one-quarter of that dataset’s train set, while FedAvg could aggregate information from its other three clients. ClientLocal is therefore not a strict single-domain upper bound.', '',
              'Per-client checkpoints and exposure records: `checkpoints/clientlocal_4datasets_seed42/client_XX/`. Per-client logs: `logs/clientlocal_4datasets_seed42/client_XX.log`. The one-client smoke run is isolated in `checkpoints/clientlocal_4datasets_seed42_smoke/`.',
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (OUT / 'summary.json').write_text(json.dumps({
        'seed': 42, 'initial_lora_sha256': initial_hash,
        'local_dataset_average': means, 'clientlocal_macro': macro,
        'fedavg_dataset_skewed_round10_macro': fed['macro_accuracy'],
        'clientlocal_minus_fedavg_macro_pp': 100*(macro-fed['macro_accuracy']),
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for domain in DOMAINS:
        print(f'Local-{domain.capitalize()}: {100*means[domain]:.2f}%')
    print(f'ClientLocal Macro: {100*macro:.2f}%')
    print(f'FedAvg-Skewed Macro: {100*fed["macro_accuracy"]:.2f}%')
    print(f'ClientLocal - FedAvg-Skewed Macro: {100*(macro-fed["macro_accuracy"]):+.2f} percentage points')
    print('ClientLocal experiment completed.')


def all_runs():
    manifest()
    LOGS.mkdir(parents=True, exist_ok=True)
    if not (SMOKE / 'client_00/completed.txt').is_file():
        with (LOGS / 'smoke.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--job', 'smoke'],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    for ci in range(16):
        if (OUT / f'client_{ci:02d}' / 'completed.txt').is_file():
            continue
        with (LOGS / f'client_{ci:02d}.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--job', 'client', '--client', str(ci)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    summarize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', choices=('all', 'smoke', 'client', 'report'), default='all')
    parser.add_argument('--client', type=int, default=None)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.job == 'all':
        all_runs()
    elif args.job == 'smoke':
        run_client(0, smoke=True)
    elif args.job == 'client':
        if args.client is None or not 0 <= args.client < 16:
            raise ValueError('--client must be 0..15')
        run_client(args.client)
    else:
        summarize()
