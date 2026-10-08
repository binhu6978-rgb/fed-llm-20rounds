"""Evaluate saved round 1-10 LoRA checkpoints and report best-test rounds.

The user requested the highest official-test Macro round as the final result.
This is a test-selected estimate and must be labelled as such.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from scripts.run_heterogeneous_mcq_4datasets import DOMAINS, RUNS, configuration, read_jsonl

METHODS = ('centralized', 'iid_mixed', 'dataset_skewed')
OUT = RUNS / 'round_diagnostics'
REPORT = ROOT / 'reports/heterogeneous_mcq_4datasets_seed42_round_curves.md'
MAIN_REPORT = ROOT / 'reports/heterogeneous_mcq_4datasets_seed42_results.md'
BEST_REPORT = ROOT / 'reports/heterogeneous_mcq_4datasets_seed42_best_round_results.md'
FIXED_REPORT = ROOT / 'reports/heterogeneous_mcq_4datasets_seed42_fixed_round10_results.md'


def checkpoint(method, round_number):
    paths = list((RUNS / method / 'adapter').glob(f'*/round_{round_number:03d}/lora_weights.pt'))
    if len(paths) != 1:
        raise RuntimeError(f'{method} round {round_number}: expected one saved adapter, found {len(paths)}')
    return paths[0]


def evaluate_method(method):
    import torch
    from peft import get_peft_model
    from utils.mcq_eval import MCQEvaluator
    from utils.model_utils import load_lora_config, load_model, load_tokenizer
    from utils.seed_utils import set_global_seed

    config = configuration(method)
    config['suffix'] = str(OUT / method)
    args = SimpleNamespace(**config)
    set_global_seed(42, device=0, deterministic=True)
    model = get_peft_model(load_model(args), load_lora_config(args))
    evaluator = MCQEvaluator(args, load_tokenizer(args))
    metrics_path = OUT / method / 'evaluation/metrics.jsonl'
    existing = {row['round'] for row in read_jsonl(metrics_path)} if metrics_path.is_file() else set()
    for round_number in range(1, 11):
        if round_number in existing:
            continue
        state = torch.load(checkpoint(method, round_number), map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=False)
        evaluator.evaluate(model, round_number)
        print(f'{method}: evaluated round {round_number}/10', flush=True)
    rows = read_jsonl(metrics_path)
    assert len(rows) == 10 and [row['round'] for row in rows] == list(range(1, 11))
    formal = read_jsonl(RUNS / method / 'evaluation/metrics.jsonl')
    assert len(formal) == 1 and formal[0]['round'] == 10
    for metric in [d + '_accuracy' for d in DOMAINS] + ['macro_accuracy', 'worst_domain_accuracy']:
        if abs(rows[-1][metric] - formal[0][metric]) > 1e-10:
            raise RuntimeError(f'Round 10 differs from formal evaluation: {method} {metric}')
    return rows


def report(curves):
    formal = json.loads((RUNS / 'final_results.json').read_text(encoding='utf-8'))
    labels = {'centralized': 'Centralized-All', 'iid_mixed': 'FedAvg-IID-Mixed',
              'dataset_skewed': 'FedAvg-Dataset-Skewed'}
    metrics = [d + '_accuracy' for d in DOMAINS] + ['macro_accuracy', 'worst_domain_accuracy']
    lines = [
        '# Four-dataset MCQ round-by-round accuracy (seed 42)', '',
        'These accuracies were computed retrospectively from saved LoRA checkpoints **after all training finished**. No model was retrained or data repartitioned. Following the updated user instruction, each method’s final result is its highest **four-dataset Macro Accuracy** over rounds 1–10. Because the official test itself selects the round, these final scores are optimistically biased and should not be presented as an unbiased held-out estimate.', '',
        '| Method | Round | MedQA | LogiQA | OpenBookQA | SciQ | Macro | Worst |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for method in METHODS:
        for row in curves[method]:
            lines.append(f"| {labels[method]} | {row['round']} | " +
                         ' | '.join(f'{100*row[name]:.2f}%' for name in metrics) + ' |')
    lines += ['', '## Highest observed official-test Macro', '',
              '| Method | Best in rounds 1–9 | Selected best in rounds 1–10 | Fixed round-10 Macro |',
              '| --- | --- | --- | ---: |']
    for method in METHODS:
        best9 = max(curves[method][:9], key=lambda row: (row['macro_accuracy'], -row['round']))
        best10 = max(curves[method], key=lambda row: (row['macro_accuracy'], -row['round']))
        fixed = formal[method]['macro_accuracy']
        lines.append(f"| {labels[method]} | R{best9['round']}: {100*best9['macro_accuracy']:.2f}% | "
                     f"R{best10['round']}: {100*best10['macro_accuracy']:.2f}% | {100*fixed:.2f}% |")
    lines += ['', f"Base Macro (one evaluation): **{100*formal['base']['macro_accuracy']:.2f}%**.", '',
              'Every retrospective round-10 result was checked against its original fixed-round result and matched exactly. Per-question predictions and metric JSONL files are in `exp/heterogeneous_mcq_4datasets/round_diagnostics/<method>/evaluation/`.']
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    selected = {method: max(curves[method], key=lambda row: (row['macro_accuracy'], -row['round']))
                for method in METHODS}
    names = [d + '_accuracy' for d in DOMAINS] + ['macro_accuracy', 'worst_domain_accuracy']
    summary = {
        'selection': 'Highest official-test four-dataset Macro Accuracy among rounds 1-10, independently per method',
        'selection_bias': 'The official test selects the checkpoint; selected scores are optimistically biased.',
        'base': formal['base'],
        'selected': selected,
        'fixed_round10': {method: formal[method] for method in METHODS},
    }
    output = RUNS / 'best_round_results.json'
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    best_lines = [
        '# Four-dataset MCQ final results: best official-test round (seed 42)', '',
        'Each trained method uses the checkpoint with the highest four-dataset Macro Accuracy among its ten saved rounds, as requested. Base is evaluated once. This is **test-set checkpoint selection**: the reported accuracies and gaps are optimistic and are not an unbiased held-out performance estimate. The original fixed-round-10 report is preserved separately.', '',
        '| Method | Selected round | MedQA | LogiQA | OpenBookQA | SciQ | Macro | Worst |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for method, label in (('base', 'Base'), ('centralized', 'Centralized-All'),
                          ('iid_mixed', 'FedAvg-IID-Mixed'), ('dataset_skewed', 'FedAvg-Dataset-Skewed')):
        row = formal['base'] if method == 'base' else selected[method]
        chosen = '—' if method == 'base' else str(row['round'])
        best_lines.append(f'| {label} | {chosen} | ' +
                          ' | '.join(f'{100*row[name]:.2f}%' for name in names) + ' |')
    iid, skew, central = (selected[method] for method in ('iid_mixed', 'dataset_skewed', 'centralized'))
    best_lines += ['', '## Differences between selected best rounds', '',
                   '| Comparison | Difference (percentage points) |', '| --- | ---: |']
    for name in [d + '_accuracy' for d in DOMAINS] + ['macro_accuracy']:
        best_lines.append(f"| IID − Dataset-Skewed: {name.replace('_accuracy', '')} | {100*(iid[name]-skew[name]):+.4f} |")
    best_lines.append(f"| Centralized − IID: macro | {100*(central['macro_accuracy']-iid['macro_accuracy']):+.4f} |")
    best_lines.append(f"| Centralized − Dataset-Skewed: macro | {100*(central['macro_accuracy']-skew['macro_accuracy']):+.4f} |")
    best_lines += ['', 'The methods may be selected at different rounds. Thus these differences compare test-selected checkpoints, not a common fixed training step. A positive IID−Skewed gap is an association in this setting, not proof of forgetting or a specific knowledge-conflict mechanism.', '',
                   'Full round-by-round accuracies: `reports/heterogeneous_mcq_4datasets_seed42_round_curves.md`. Original fixed-round-10 report: `reports/heterogeneous_mcq_4datasets_seed42_fixed_round10_results.md`. Source partitions: `dataset/heterogeneous_mcq_4datasets/manifest.json`.']
    best_text = '\n'.join(best_lines) + '\n'
    if MAIN_REPORT.is_file() and not FIXED_REPORT.is_file():
        shutil.copyfile(MAIN_REPORT, FIXED_REPORT)
    BEST_REPORT.write_text(best_text, encoding='utf-8')
    MAIN_REPORT.write_text(best_text, encoding='utf-8')
    print(f'Round curves report: {REPORT}', flush=True)
    print(f'Best-round final report: {MAIN_REPORT}', flush=True)


def main(wait_for_final):
    final_path = RUNS / 'final_results.json'
    if wait_for_final:
        print('Waiting until the formal fixed-round experiment has finished.', flush=True)
        deadline = time.monotonic() + 48 * 3600
        while not final_path.is_file():
            if time.monotonic() > deadline:
                raise TimeoutError('Formal experiment did not complete within 48 hours')
            time.sleep(30)
    elif not final_path.is_file():
        raise RuntimeError('Formal experiment is not complete; pass --wait-for-final')
    curves = {method: evaluate_method(method) for method in METHODS}
    report(curves)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-for-final', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    main(args.wait_for_final)
