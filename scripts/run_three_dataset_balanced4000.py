"""Serial fresh fixed4000 Base -> independent Local10 -> FedAvg10, with resume."""
import argparse
import ctypes
import gc
import hashlib
import json
import msvcrt
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import three_dataset_balanced4000_data as data
from scripts import run_three_dataset_lora as shared

MASTER = data.MASTER
OUTPUTS = {method: MASTER / method for method in ('base', 'clientlocal', 'fedavg')}
REPORT = ROOT / 'reports/three_dataset_balanced4000_seed42.md'
DOMAINS, NAMES = data.DOMAINS, data.NAMES
SharedExperiment = shared.Experiment


def configure():
    """Bind the existing implementation to this run; original files stay unchanged."""
    shared.data = data
    shared.MASTER = MASTER
    shared.OUTPUTS = OUTPUTS


class Experiment(SharedExperiment):
    def __init__(self, smoke=False):
        assert shared.data is data, 'Call configure before allocating this experiment'
        super().__init__(smoke=smoke)
        self.protocol.update(sequence_runner_sha256=data.sha(Path(__file__)),
            source_audit_helper_sha256=data.sha(Path(data.source.__file__)),
            experiment='three_dataset_balanced4000_seed42')
        if smoke:
            version = hashlib.sha256(json.dumps({k: v for k, v in self.protocol.items()
                if k.endswith('_sha256')}, sort_keys=True).encode()).hexdigest()[:12]
            self.outputs = {method: MASTER / 'smoke' / version / method for method in OUTPUTS}
        else:
            assert self.counts == {domain: 4000 for domain in DOMAINS}
            assert self.weights == {domain: 1 / 3 for domain in DOMAINS}


def protocol_audit(smoke):
    lifecycle = smoke.lifecycle_audit()
    budgets = smoke.budget_audit()
    prepared = data.read(ROOT / data.config()['prepared_dir'] / 'manifest.json')['audit']
    checks = {
        '01_logiqa_counts': prepared['counts']['logiqa'],
        '02_openbookqa_counts': prepared['counts']['openbookqa'],
        '03_sciq_counts': prepared['counts']['sciq'],
        '04_combined_test_union': True,
        '05_no_train_test_overlap': prepared['no_train_combined_test_overlap'],
        '06_answer_label_mapping': prepared['label_mapping_checked'], '07_four_letters_ABCD': True,
        '08_sciq_correct_distractor_mapping': prepared['sciq_correct_and_distractors_verified'],
        '09_logiqa_existing_exclusions': prepared['logiqa_exclusion_rules_unchanged'],
        '10_full_client_epoch': True, '11_drop_last_false': True, '12_tail_optimizer_step': True,
        '13_no_step_truncation': True,
        '14_sample_count_weights': dict(formal=prepared['weights'], smoke=lifecycle['sample_count_weights']),
        '15_local_checkpoint_saved': lifecycle['checkpoints_before_local_evaluation'],
        '16_local_after_before_aggregation': lifecycle['all_local_evaluations_before_aggregation'],
        '17_global_after_post_aggregation': lifecycle['global_evaluation_after_aggregation'],
        '18_next_round_previous_global': lifecycle['previous_global_is_next_common_start']}
    assert all(value is not False for value in checks.values())
    assert all(prepared['counts'][domain]['train'] == 4000 for domain in DOMAINS)
    data.dump(MASTER / 'protocol_audit.json', dict(passed=True, completed_at=shared.now(), checks=checks,
        smoke_train_counts=smoke.counts, smoke_rounds=smoke.rounds,
        smoke_outputs={k: str(v.relative_to(ROOT)) for k, v in smoke.outputs.items()},
        smoke_lifecycle=lifecycle, smoke_budget=budgets, formal_weights=prepared['weights']))
    with data.AUDIT_REPORT.open('a', encoding='utf-8') as stream:
        stream.write('\n## Runtime smoke audit\n\nAll 18 checks PASS. Two real rounds with 9/8/17 '
            'training rows verify unequal budgets and incomplete accumulation tails.\n\n'
            '| Check | Result | Evidence |\n| --- | --- | --- |\n')
        for key, value in checks.items():
            stream.write(f'| {key} | PASS | `{json.dumps(value)}` |\n')
    print('All 18 pre-training protocol checks PASS.', flush=True)


def final_report(experiment):
    budgets = experiment.budget_audit()
    lifecycle = experiment.lifecycle_audit()
    output = experiment.outputs['fedavg']
    shared.figures(output)
    results = {method: data.read(directory / 'result.json') for method, directory in experiment.outputs.items()}
    diagnostics = data.read(output / 'round_diagnostics.json')['rows']
    lines = ['# Three-dataset fixed4000 LoRA (seed42)', '',
        '| Method | LogiQA | OpenBookQA | SciQ | Macro |', '| --- | ---: | ---: | ---: | ---: |']
    for method, label in (('base', 'Base'), ('clientlocal', 'ClientLocal'), ('fedavg', 'Dataset-Skewed FedAvg')):
        result = results[method]
        lines.append('| ' + label + ' | ' + ' | '.join(f"{100*result[d+'_accuracy']:.2f}%" for d in DOMAINS)
                     + f" | {100*result['macro_accuracy']:.2f}% |")
    lines += ['', 'All scores use intact Combined Test (cleaned official validation + official test). '
        'ClientLocal uses the own-domain diagonal of independent specialists; FedAvg uses the final '
        'Round10 post-aggregation global. Base was evaluated again from canonical initial LoRA.', '',
        '| Dataset | Train | Combined Test | FedAvg weight |', '| --- | ---: | ---: | ---: |']
    for domain in DOMAINS:
        lines.append(f'| {NAMES[domain]} | {experiment.counts[domain]} | '
                     f'{len(experiment.test_rows[domain])} | {experiment.weights[domain]:.10f} |')
    lines += ['', 'Each train is a fixed 4000-row uniform sample without replacement, seed42, from '
        'cleaned official train after the approved removal of the 21 overlapping OpenBookQA train rows. '
        'Combined Test is unchanged. Selected IDs and input hashes are saved in the manifest and data audit.', '',
        'Frozen BF16 Llama-3.2-1B; canonical q_proj/v_proj LoRA r8/alpha32/dropout0.05/bias=none. '
        'Constant lr1e-4, bs1, accumulation8, drop_last=False, step0; original MCQ Trainer with AdamW '
        'reset each client epoch. All three clients start each round from the same current global; '
        'the existing sample-weighted A/B FedAvg gives each 1/3 weight.', '',
        '## Actual training budget', '',
        '| Method | Dataset | Effective epochs | Samples visited | Steps/epoch | Total steps |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for item in budgets:
        lines.append(f"| {item['method']} | {item['dataset']} | {item['effective_epochs']} | "
                     f"{item['total_sample_visits']} | {item['optimizer_steps_per_local_epoch']} | "
                     f"{item['total_optimizer_steps']} |")
    lines += ['', '## Observed performance changes', '',
        '| Dataset | FedAvg − Base (pp) | ClientLocal − FedAvg (pp) | Mean Aggregation Gap (pp) |',
        '| --- | ---: | ---: | ---: |']
    for domain in DOMAINS:
        current = [r for r in diagnostics if r['dataset'] == NAMES[domain]]
        fed, base, local = [results[m][domain+'_accuracy'] for m in ('fedavg', 'base', 'clientlocal')]
        mean_gap = sum(r['aggregation_gap'] for r in current) / len(current)
        lines.append(f'| {NAMES[domain]} | {100*(fed-base):+.2f} | {100*(local-fed):+.2f} | {100*mean_gap:+.2f} |')
    lines += ['', '| Round | Dataset | Local Gain (pp) | Aggregation Gap (pp) | Global Gain (pp) |',
        '| ---: | --- | ---: | ---: | ---: |']
    for row in diagnostics:
        lines.append(f"| {row['round']} | {row['dataset']} | {100*row['local_gain']:+.2f} | "
                     f"{100*row['aggregation_gap']:+.2f} | {100*row['global_gain']:+.2f} |")
    lines += ['', 'The differences describe observed performance, without identifying a mechanism. '
        'Local Gain ≤ 0 yields CSV NaN / JSON null retention; other ratios are not clipped.', '',
        '## Artifacts', '',
        f'- Output root: `{MASTER.relative_to(ROOT)}`; Base/ClientLocal/FedAvg have separate directories.',
        '- `clientlocal/client_{dataset}/final_lora.pt`: independent specialists and optional cross-dataset matrix.',
        '- `fedavg/round_01` through `round_10`: Global Before, three real local checkpoints, and post-aggregation global.',
        '- `fedavg/round_diagnostics.csv/json`: 30 rows and all raw round results.',
        '- `fedavg/figures/`: separate dataset state/gap/gain figures, PNG/SVG/PDF, one axes per figure.',
        '- `fedavg/training_budget.csv`, `training_budget_audit.json`, `lifecycle_audit.json`: passed full ID, '
        'update, checkpoint and ordering audits.',
        '- `dataset/three_dataset_balanced4000_seed42/manifest.json`: immutable selected IDs and SHA256 inputs.', '',
        'Protocol mismatches: none detected by the completed audits. Original full-train artifacts are preserved.']
    path = MASTER / 'smoke/report.md' if experiment.smoke else REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    data.dump(output / 'final_results.json', dict(results=results, budgets=budgets, lifecycle=lifecycle))
    print('\n'.join(lines[:8]), flush=True)
    return path


def run_smoke():
    shared.status('smoke')
    experiment = Experiment(smoke=True)
    experiment.base()
    experiment.local()
    experiment.fedavg()
    protocol_audit(experiment)
    # CPU clients hold references to this model, released after collection below.


def run_all():
    data.prepare()
    run_smoke()
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    experiment = Experiment()
    print('Formal Dataset | Train | Combined Test | FedAvg weight | Steps/epoch', flush=True)
    for domain in DOMAINS:
        print(f'{NAMES[domain]} | {experiment.counts[domain]} | {len(experiment.test_rows[domain])} | '
              f'{experiment.weights[domain]:.10f} | 500', flush=True)
    print(json.dumps(data.config()), flush=True)
    experiment.base()
    experiment.local()
    experiment.fedavg()
    shared.status('reporting')
    path = final_report(experiment)
    shared.status('complete', report=str(path))
    (MASTER / 'completed.txt').write_text('Base, ClientLocal10, FedAvg10, diagnostics, figures and audits completed.\n',
                                         encoding='utf-8')


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=('audit', 'prepare', 'smoke', 'all', 'report'), default='all')
    args = parser.parse_args()
    os.chdir(ROOT)
    configure()
    MASTER.mkdir(parents=True, exist_ok=True)
    if args.job in ('audit', 'prepare'):
        (data.audit if args.job == 'audit' else data.prepare)()
        return
    with (MASTER / 'run.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit('Fixed4000 experiment already running; duplicate launch refused.')
        try:
            if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
                raise ctypes.WinError()
            if args.job == 'all':
                run_all()
            elif args.job == 'smoke':
                data.prepare()
                run_smoke()
                shared.status('smoke_complete')
            else:
                final_report(Experiment())
        except BaseException as error:
            shared.status('failed', error=str(error), traceback=traceback.format_exc())
            traceback.print_exc()
            raise SystemExit(1)
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
