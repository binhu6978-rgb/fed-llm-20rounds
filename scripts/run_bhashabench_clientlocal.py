"""One smoke run, then 12 independent ten-epoch LoRA runs on the frozen partition."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

DATA = ROOT / 'dataset/bhashabench_mcq'
INITIAL = ROOT / 'exp/bhashabench_llama32/domain_skewed/initial_lora.pt'
OUT = ROOT / 'exp/bhashabench_llama32/clientlocal'
LOGS = ROOT / 'logs/bhashabench_llama32/clientlocal'
CONFIG = ROOT / 'configs/bhashabench_mcq_clientlocal.yaml'
REPORT = ROOT / 'reports/bhashabench_llama32_clientlocal_seed42_results.md'
CURVES = ROOT / 'reports/bhashabench_llama32_clientlocal_seed42_curves.csv'
DOMAINS = ('ayur', 'legal', 'krishi')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def manifest_and_config():
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    files = manifest['conditions']['domain_skewed']
    assert len(files) == 12
    for ci, entry in enumerate(files):
        path = ROOT / entry['path']
        assert entry['id'] == ci and entry['rows'] == 504 and sha(path) == entry['sha256']
        assert entry['domain_counts'][DOMAINS[ci // 4]] == 504
    for domain in DOMAINS:
        path = DATA / 'test' / f'{domain}.jsonl'
        assert len(manifest['test_files'][domain]['ids']) == 400
        assert sha(path) == manifest['test_files'][domain]['sha256']
    assert INITIAL.is_file()
    config = yaml.safe_load((ROOT / 'configs/bhashabench_mcq_domain_skewed.yaml').read_text(encoding='utf-8'))
    config.update(alg='clientlocal', suffix='bhashabench_llama32/clientlocal',
                  clientlocal_epochs=10, initial_adapter_path=str(INITIAL),
                  note='Run via scripts/run_bhashabench_clientlocal.py; no communication or FedAvg.')
    CONFIG.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
    return manifest, config


def train(smoke=False):
    from alg.clientlocal import Client, Server
    from utils.seed_utils import set_global_seed
    import torch

    _, config = manifest_and_config()
    config['suffix'] = str(OUT) if not smoke else str(OUT.parent / 'clientlocal_smoke')
    args = SimpleNamespace(**config)
    set_global_seed(42, device=0, deterministic=True)
    clients = [Client(ci, args) for ci in range(12)]
    if smoke:
        for client in clients:
            dataset = client.dataset['train']
            dataset.rows = dataset.rows[:8]
            dataset.encoded = dataset.encoded[:8]
    set_global_seed(42, device=0, deterministic=True)
    server = Server(args, clients, INITIAL)
    evaluator = server.global_evaluator
    if smoke:
        for domain in DOMAINS:
            evaluator.rows[domain] = evaluator.rows[domain][:2]
            evaluator.encoded[domain] = evaluator.encoded[domain][:2]
    all_rows = []
    epochs = 1 if smoke else 10
    for client in clients:
        domain = DOMAINS[client.id // 4]
        output = Path(args.suffix) / f'client_{client.id:02d}'
        evaluator.output = output / 'evaluation'
        evaluator.output.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, epochs + 1):
            loss = server.train_client_epoch(client, epoch)
            metrics = evaluator.evaluate(server.model, epoch)
            row = {'client': client.id, 'train_domain': domain, 'epoch': epoch,
                   'train_loss': loss, 'train_samples': client.last_seen_count,
                   'optimizer_updates': client.last_optimizer_updates,
                   'ayur_accuracy': metrics['ayur_accuracy'],
                   'legal_accuracy': metrics['legal_accuracy'],
                   'krishi_accuracy': metrics['krishi_accuracy'],
                   'macro_accuracy': metrics['macro_accuracy'],
                   'own_domain_accuracy': metrics[domain + '_accuracy']}
            all_rows.append(row)
            print(f"Client {client.id} Epoch {epoch}: own={row['own_domain_accuracy']:.6f} "
                  f"macro={row['macro_accuracy']:.6f} loss={loss:.6f}", flush=True)
        checkpoint = server.save_client_adapter(client.id)
        if smoke:
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert all(torch.equal(saved[k], server.client_loras[client.id][k]) for k in saved)
            for other in range(client.id + 1, 12):
                assert all(torch.equal(server.client_loras[other][k], server.initial_lora[k])
                           for k in server.initial_lora), 'Untrained client inherited another client state'
        (output / 'completed.txt').write_text(f'Client {client.id}: {epochs} full epochs completed.\n', encoding='utf-8')
    path = Path(args.suffix) / 'curves.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (Path(args.suffix) / 'completed.txt').write_text(f'{len(clients)} clients × {epochs} epochs completed.\n', encoding='utf-8')
    print(f'Completed {len(clients)} independent clients × {epochs} epochs.', flush=True)


def report():
    from utils.mcq_utils import read_jsonl
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    with (OUT / 'curves.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 120
    source_initial_sha = sha(INITIAL)
    for ci in range(12):
        subset = [r for r in rows if int(r['client']) == ci]
        assert [int(r['epoch']) for r in subset] == list(range(1, 11))
        original = read_jsonl(ROOT / manifest['conditions']['domain_skewed'][ci]['path'])
        expected = {r['domain'] + ':' + r['id'] for r in original}
        exposure = [json.loads(line) for line in (OUT / 'exposure.jsonl').read_text(encoding='utf-8').splitlines()
                    if line.strip()]
        relevant = [r for r in exposure if r['client'] == ci]
        assert len(relevant) == 10
        assert all(len(r['ids']) == len(set(r['ids'])) == 504 and set(r['ids']) == expected
                   and r['optimizer_updates'] == 63 for r in relevant)
        assert (OUT / f'client_{ci:02d}' / 'lora_weights.pt').is_file()
        for ep in range(1, 11):
            pred = OUT / f'client_{ci:02d}' / 'evaluation' / f'round_{ep:02d}_predictions.csv'
            with pred.open(encoding='utf-8', newline='') as handle:
                assert sum(1 for _ in csv.DictReader(handle)) == 1200
    old = json.loads((ROOT / 'exp/bhashabench_llama32/final_results.json').read_text(encoding='utf-8'))
    final = [r for r in rows if int(r['epoch']) == 10]
    means = {domain: sum(float(r['own_domain_accuracy']) for r in final if r['train_domain'] == domain) / 4
             for domain in DOMAINS}
    local_macro = sum(means.values()) / 3
    lines = [
        '# BhashaBench × Llama-3.2-1B ClientLocal（seed 42）', '',
        '## 数据与协议', '',
        '完全复用 `dataset/bhashabench_mcq/manifest.json` 中 Domain-Skewed 的 client 0–11：各 504 条训练题；test 为每域原有 400 条。没有重新划分、清洗或修改冻结池。源 partition 文件的 SHA256 与原 manifest 一致。', '',
        f'12 个独立 LoRA 均从同一个 Llama-3.2-1B base 和 Domain-Skewed 正式实验的同一初始 LoRA 开始。初始 adapter SHA256：`{source_initial_sha}`。共享模型容器仅用于顺序执行；每次加载该客户端自己的 LoRA 状态，客户端之间没有通信或参数聚合。', '',
        'BF16 base，LoRA q_proj/v_proj，r=8、alpha=32、dropout=0.05、bias=none；只训练 LoRA。seed=42，常数 lr=1e-4，batch=1、grad_accum=8。每 client 10 个完整 epoch，每 epoch 504 samples/63 updates，每题访问 10 次。与此前 FedAvg 每轮重置 optimizer 的实现一致，这里每 epoch 也新建 AdamW；不是跨 10 epoch 持久化的 Adam 动量。', '',
        'prompt、`" A"/" B"/" C"/" D"` continuation likelihood 和无生成评价完全复用原 MCQ protocol。每个 local model 每个 epoch 都在相同的三个 test domain 上测评；下表主结果固定 epoch 10，没有选择最佳 epoch。', '',
        '## 每客户端结果：Epoch 10', '',
        '| Client | Train Domain | Ayur | Legal | Krishi | Macro | Own-domain |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for r in final:
        vals = [r['ayur_accuracy'], r['legal_accuracy'], r['krishi_accuracy'],
                r['macro_accuracy'], r['own_domain_accuracy']]
        lines.append(f"| {r['client']} | {r['train_domain']} | "
                     + ' | '.join(f'{100 * float(v):.2f}%' for v in vals) + ' |')
    lines += ['', '## 每领域 Local own-domain 平均', '',
              '| Domain | Local clients | Avg own-domain accuracy |',
              '| --- | --- | ---: |']
    for di, domain in enumerate(DOMAINS):
        lines.append(f'| {domain.capitalize()} | {4 * di}–{4 * di + 3} | {100 * means[domain]:.2f}% |')
    lines += ['', f'**ClientLocal own-domain Macro：{100 * local_macro:.2f}%**', '',
              '## 与第一阶段结果并列', '',
              '| Method | Score | 口径 |', '| --- | ---: | --- |',
              f"| Base | {100 * old['base']['macro_accuracy']:.2f}% | 单一共享模型的三域 macro |",
              f"| Centralized-All | {100 * old['centralized']['macro_accuracy']:.2f}% | 单一共享模型的三域 macro |",
              f"| IID-Mixed | {100 * old['iid_mixed']['macro_accuracy']:.2f}% | 单一共享模型的三域 macro |",
              f"| Domain-Skewed FedAvg | {100 * old['domain_skewed']['macro_accuracy']:.2f}% | 单一共享模型的三域 macro |",
              f'| ClientLocal own-domain | {100 * local_macro:.2f}% | 12 个模型各自在其训练领域的 accuracy 均值 |',
              '',
              '**口径提醒：**ClientLocal own-domain 使用领域已知的本地模型，等权平均三个领域各四个客户端的本域准确率；它不是一个共享模型在三域上的 macro，表中分数并非同一个评价对象的直接排名。每个 local model 的完整三域结果见上表。', '',
              '## Epoch 1–10 local accuracy curves', '',
              '| Client | Epoch | Ayur | Legal | Krishi | Macro | Own-domain | Train loss |',
              '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
              ]
    for r in rows:
        acc = [r['ayur_accuracy'], r['legal_accuracy'], r['krishi_accuracy'],
               r['macro_accuracy'], r['own_domain_accuracy']]
        lines.append(f"| {r['client']} | {r['epoch']} | "
                     + ' | '.join(f'{100 * float(v):.2f}%' for v in acc)
                     + f" | {float(r['train_loss']):.6f} |")
    lines += ['', '## 文件与复现', '',
              '- 本地模型、LoRA 超参数、MCQ 协议和原 partition/test：沿用第一阶段配置与 manifest。',
              '- 新配置：`configs/bhashabench_mcq_clientlocal.yaml`；运行脚本：`scripts/run_bhashabench_clientlocal.py`。',
              '- 每 client 的 adapter 与逐题 epoch 1–10 predictions/logprobs：`exp/bhashabench_llama32/clientlocal/client_XX/`。',
              '- 机器可读的曲线：`reports/bhashabench_llama32_clientlocal_seed42_curves.csv`；完整 exposure：`exp/bhashabench_llama32/clientlocal/exposure.jsonl`；日志：`logs/bhashabench_llama32/clientlocal/`。',
              '',
              '只比较当前 BhashaBench-derived 三领域 MCQ 和单 seed 的观察结果，不由此单独证明知识遗忘或普遍的跨域机制。']
    CURVES.write_text((OUT / 'curves.csv').read_text(encoding='utf-8'), encoding='utf-8')
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (OUT / 'final_summary.json').write_text(json.dumps({
        'seed': 42, 'initial_lora_sha256': source_initial_sha,
        'local_own_domain_average': means, 'local_own_domain_macro': local_macro,
        'domain_skewed_fedavg_shared_macro': old['domain_skewed']['macro_accuracy'],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Ayur Local Avg: {100 * means["ayur"]:.2f}%')
    print(f'Legal Local Avg: {100 * means["legal"]:.2f}%')
    print(f'Krishi Local Avg: {100 * means["krishi"]:.2f}%')
    print(f'ClientLocal own-domain Macro: {100 * local_macro:.2f}%')
    print(f'Domain-Skewed FedAvg Macro: {100 * old["domain_skewed"]["macro_accuracy"]:.2f}%')
    print('Experiment completed.')


def all_runs():
    LOGS.mkdir(parents=True, exist_ok=True)
    manifest_and_config()
    for job in ('smoke', 'full'):
        completed = (OUT.parent / 'clientlocal_smoke' if job == 'smoke' else OUT) / 'completed.txt'
        if completed.is_file():
            continue
        with (LOGS / f'{job}.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--job', job],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', choices=('all', 'smoke', 'full', 'report'), default='all')
    job = parser.parse_args().job
    os.chdir(ROOT)
    if job == 'all':
        all_runs()
    elif job == 'smoke':
        train(smoke=True)
    elif job == 'full':
        train(smoke=False)
    else:
        report()
