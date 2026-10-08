"""Retrospectively evaluate all saved centralized epochs and inventory Local maxima."""
import argparse
import ctypes
import csv
import json
import msvcrt
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_cosmosqa_five_experiments as runner
from scripts import run_three_dataset_lora as shared
from scripts import cosmosqa_five_data as data

DOMAINS, NAMES = data.DOMAINS, data.NAMES
OUTPUT = data.MASTER / 'best_saved_analysis'
REPORT = ROOT / 'reports/cosmosqa_five_best_saved_seed42.md'


def status(stage, **details):
    data.dump(OUTPUT / 'status.json', dict(stage=stage, updated_at=shared.now(), **details))


def local_inventory():
    import torch
    candidates, selected = {}, {}
    for d in DOMAINS:
        _, folder = data.baseline_sources(d)
        checkpoint = folder / 'final_lora.pt'
        metrics = data.read(folder / 'result.json')
        receipt = data.read(folder / 'evaluation_provenance.json')
        assert shared.tensor_hash(torch.load(checkpoint, map_location='cpu', weights_only=True)) == receipt['lora_tensor_sha256']
        choices = [dict(epoch=10, accuracy=metrics[d + '_accuracy'], checkpoint=str(checkpoint),
                        metrics=str(folder / 'result.json'), receipt=str(folder / 'evaluation_provenance.json'))]
        if d == 'cosmosqa':
            root = ROOT / 'outputs/cosmosqa_balanced4000_seed42/diagnostics'
            for path in sorted(root.glob('epoch_*/summary.json')):
                value = data.read(path)
                e = value['epoch']
                if e == 10: continue
                ckpt = Path(value['checkpoint'])
                proof = data.read(path.parent / 'evaluation_provenance.json')
                capture = data.read(path.parent / 'capture_receipt.json')
                assert capture['epoch'] == e
                assert shared.tensor_hash(torch.load(ckpt, map_location='cpu', weights_only=True)) == proof['lora_tensor_sha256'] == capture['lora_tensor_sha256']
                choices.append(dict(epoch=e, accuracy=value['accuracy'], checkpoint=str(ckpt),
                    metrics=str(path), receipt=str(path.parent / 'evaluation_provenance.json')))
        choices.sort(key=lambda row: row['epoch'])
        candidates[d] = choices
        selected[d] = dict(max(choices, key=lambda row: (row['accuracy'], -row['epoch'])),
            available_epochs=[row['epoch'] for row in choices], complete_ten_epoch_coverage=len(choices) == 10)
    result = dict(candidates=candidates, selected=selected,
        macro_accuracy=sum(selected[d]['accuracy'] for d in DOMAINS) / 5,
        complete_ten_epoch_coverage=all(v['complete_ten_epoch_coverage'] for v in selected.values()),
        interpretation='Best among saved Independent Local checkpoints only; no missing epochs imputed')
    data.dump(OUTPUT / 'local_inventory.json', result)
    return result


def central_summary(records):
    assert [r['epoch'] for r in records] == list(range(1, 11))
    # One shared checkpoint, selected on the same fixed five-domain Macro.
    best = max(records, key=lambda r: (r['metrics']['macro_accuracy'], -r['epoch']))
    return dict(selection_metric='macro_accuracy on existing fixed test', tie_break='earliest epoch',
        selected_epoch=best['epoch'], selected_checkpoint=best['checkpoint'], selected_metrics=best['metrics'],
        complete_ten_epoch_coverage=True, epochs=records, retrospective_test_selected=True)


def report(central, local):
    baseline = data.read(data.MASTER / 'base/result.json')
    endpoint_local = data.read(data.MASTER / 'clientlocal/result.json')
    fed = data.read(data.MASTER / 'fedavg/result.json')
    central10 = data.read(data.MASTER / 'centralized/result.json')
    best_local = {d + '_accuracy': local['selected'][d]['accuracy'] for d in DOMAINS}
    best_local['macro_accuracy'] = local['macro_accuracy']
    lines = ['# CosmosQA five-domain saved-checkpoint maxima (seed42)', '',
        'User requested retrospective best-epoch reporting after the fixed epoch10 runs completed. '
        'The original endpoint report/checkpoints remain intact. No training or resampling is performed here.', '',
        f"Centralized: all ten epochs are evaluated on the same existing fixed test; epoch {central['selected_epoch']} "
        'has the highest five-domain Macro (ties select the earliest epoch). All domain accuracies in this row '
        'come from that same shared checkpoint, not separate per-domain maxima.',
        'Local: choose the own-domain maximum among available Independent Local checkpoints. '
        'Missing Local epochs cannot be reconstructed from a loss curve; this row is an observed saved-checkpoint '
        'maximum, not the best of ten epochs.', '',
        '| Method | ' + ' | '.join(NAMES[d] for d in DOMAINS) + ' | Macro |',
        '| --- | ' + ' | '.join(['---:'] * 6) + ' |']
    methods = [('Base', baseline), ('Independent Local epoch10', endpoint_local),
        ('Independent Local highest saved (incomplete coverage)', best_local), ('FedAvg Round10', fed),
        ('Centralized Epoch10', central10), (f"Centralized highest Macro (epoch {central['selected_epoch']})", central['selected_metrics'])]
    for label, values in methods:
        lines.append('| ' + label + ' | ' + ' | '.join(f"{values[d + '_accuracy'] * 100:.2f}%" for d in DOMAINS)
                     + f" | {values['macro_accuracy'] * 100:.2f}% |")
    lines += ['', '## Centralized all ten epochs', '',
        '| Epoch | ' + ' | '.join(NAMES[d] for d in DOMAINS) + ' | Macro |',
        '| ---: | ' + ' | '.join(['---:'] * 6) + ' |']
    for record in central['epochs']:
        values = record['metrics']
        lines.append(f"| {record['epoch']} | " + ' | '.join(f"{values[d + '_accuracy'] * 100:.2f}%" for d in DOMAINS)
            + f" | {values['macro_accuracy'] * 100:.2f}% |")
    lines += ['', '## Independent Local checkpoint availability', '',
        '| Domain | Saved epochs | Highest saved epoch | Accuracy | Full ten-epoch maximum available |',
        '| --- | --- | ---: | ---: | --- |']
    for d in DOMAINS:
        value = local['selected'][d]
        lines.append(f"| {NAMES[d]} | {','.join(map(str, value['available_epochs']))} | {value['epoch']} | "
                     f"{100 * value['accuracy']:.2f}% | {value['complete_ten_epoch_coverage']} |")
    lines += ['', 'These maxima are selected on the existing test and are reported as retrospective best-observed '
        'diagnostics. The original fixed epoch10 comparison remains available separately. '
        'No validation-selected or new held-out accuracy is claimed.', '',
        f"Selected centralized checkpoint: `{central['selected_checkpoint']}`.",
        f"Predictions, four choice log probabilities and receipts: `{OUTPUT.relative_to(ROOT)}/centralized/epoch_XX/`.",
        f"Full Local candidate inventory and metrics: `{OUTPUT.relative_to(ROOT)}/local_inventory.json`."]
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    data.dump(OUTPUT / 'result.json', dict(centralized=central, local=local, report=str(REPORT)))


def main():
    import torch
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'): stream.reconfigure(encoding='utf-8')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / 'run.lock').open('a+b') as lock:
        lock.seek(0)
        try: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError: raise SystemExit('Best-saved evaluation already running')
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            inputs = data.verify()
            local = local_inventory()
            central_dir = data.MASTER / 'centralized'
            checkpoint_paths = [central_dir / f'epoch_{e:02d}_lora.pt' for e in range(1, 11)]
            assert all(path.exists() for path in checkpoint_paths)
            protected = {str(path.relative_to(ROOT)): data.sha(path) for path in checkpoint_paths}
            for name in ('result.json', 'evaluation_provenance.json', 'protocol.json', 'training_epochs.json'):
                path = central_dir / name; protected[str(path.relative_to(ROOT))] = data.sha(path)
            original_report = ROOT / 'reports/cosmosqa_five_comparison_seed42.md'
            protected[str(original_report.relative_to(ROOT))] = data.sha(original_report)
            curves = data.read(central_dir / 'training_epochs.json')
            experiment = runner.Experiment()
            records = []
            for e, checkpoint in enumerate(checkpoint_paths, 1):
                status('centralized_evaluation', epoch=e)
                state = torch.load(checkpoint, map_location='cpu', weights_only=True)
                fingerprint = shared.tensor_hash(state)
                assert fingerprint == curves[e - 1]['after_lora_tensor_sha256']
                experiment.load(state)
                if e == 10:
                    proof = data.read(central_dir / 'evaluation_provenance.json')
                    assert proof['lora_tensor_sha256'] == fingerprint
                    scores = data.read(central_dir / 'result.json')
                    directory = central_dir
                else:
                    directory = OUTPUT / 'centralized' / f'epoch_{e:02d}'
                    scores = experiment.score(directory, DOMAINS, e, checkpoint)
                records.append(dict(epoch=e, checkpoint=str(checkpoint.relative_to(ROOT)),
                    lora_tensor_sha256=fingerprint, metrics=scores, evaluation_dir=str(directory.relative_to(ROOT))))
                data.dump(OUTPUT / 'centralized_epoch_metrics.json', records)
                print(f'Centralized epoch {e}: Macro={100 * scores["macro_accuracy"]:.2f}%', flush=True)
            for name, digest in protected.items(): assert data.sha(ROOT / name) == digest, name
            data.verify()
            central = central_summary(records)
            report(central, local)
            data.dump(OUTPUT / 'audit.json', dict(passed=True, all_ten_centralized_epochs_scored=True,
                centralized_single_checkpoint_selection=True, train_performed=False, resampling=False,
                original_artifacts_unchanged=True, input_integrity=inputs,
                selected_centralized_epoch=central['selected_epoch'], local_full_ten_epoch_coverage=local['complete_ten_epoch_coverage'],
                original_sha256=protected, retrospective_test_selection=True))
            status('complete', report=str(REPORT), selected_centralized_epoch=central['selected_epoch'],
                centralized_macro_accuracy=central['selected_metrics']['macro_accuracy'],
                local_saved_macro_accuracy=local['macro_accuracy'])
        except BaseException as error:
            status('failed', error=str(error), traceback=traceback.format_exc())
            raise
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__': main()
