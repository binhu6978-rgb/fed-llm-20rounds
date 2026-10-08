"""Attach to current CosmosQA Local10, then serial smoke -> FedAvg10 -> Centralized10 -> report."""
import argparse
import ctypes
import json
import msvcrt
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cosmosqa_five_data as data
from scripts import cosmosqa_data as cosmos
from scripts import run_three_dataset_lora as shared

QUEUE = data.MASTER / 'sequence'
LOGS = ROOT / 'logs/cosmosqa_five_client4000_seed42'


def status(stage, **details):
    data.dump(QUEUE / 'status.json', dict(stage=stage, pid=os.getpid(), updated_at=shared.now(), **details))


def wait_process(pid, script):
    if not pid or not psutil.pid_exists(pid): return
    process = psutil.Process(pid)
    assert any(script in item for item in process.cmdline()), 'PID is not the expected experiment'
    assert Path(process.cwd()).resolve() == ROOT
    created = process.create_time()
    while process.is_running():
        try:
            if process.create_time() != created: break
        except psutil.NoSuchProcess: break
        time.sleep(10)


def child(stage, arguments):
    # A killed queue may leave its child alive; attach to that exact child first.
    old = data.read(QUEUE / 'status.json') if (QUEUE / 'status.json').exists() else {}
    if old.get('stage') == stage and old.get('child_pid'):
        wait_process(old['child_pid'], 'run_cosmosqa_five_experiments.py')
    path = LOGS / (stage + '.log')
    with path.open('a', encoding='utf-8') as stream:
        stream.write('\nRun ' + shared.now() + '\n'); stream.flush()
        process = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', *arguments], cwd=ROOT,
            stdout=stream, stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1'))
        status(stage, child_pid=process.pid, log=str(path))
        code = process.wait()
    if code != 0: raise RuntimeError(f'{stage} failed (exit {code}); subsequent methods not started: {path}')


def patch_cosmos_diagnostic_report():
    # Preserve the formal epoch10 comparison and disclose the separately requested diagnostics.
    report = ROOT / 'reports/cosmosqa_sanity_seed42.md'
    summaries = sorted(cosmos.MASTER.glob('diagnostics/epoch_*/summary.json'))
    if not summaries or not report.exists(): return
    text = report.read_text(encoding='utf-8')
    text = text.replace('No intermediate accuracy or checkpoint selection.',
        'User-requested intermediate accuracy diagnostics are recorded separately below; no checkpoint selection.')
    marker = '\n## User-requested intermediate diagnostics\n'
    if marker in text: text = text.split(marker)[0]
    text += marker + '\n| Epoch | Accuracy | Local − Base (pp) |\n| ---: | ---: | ---: |\n'
    values = [data.read(path) for path in summaries]
    for value in values:
        text += f"| {value['epoch']} | {value['accuracy'] * 100:.2f}% | {value['local_minus_base_pp']:+.2f} |\n"
    text += '\nThese diagnostics were requested while training was already running; no resampling, tuning or endpoint selection. Formal Local remains epoch10.\n'
    report.write_text(text, encoding='utf-8')
    audit = data.read(cosmos.MASTER / 'protocol_audit.json')
    audit['formal_accuracy_evaluation_epochs'] = audit['accuracy_evaluation_epochs']
    audit['diagnostic_accuracy_evaluation_epochs'] = [v['epoch'] for v in values]
    audit['accuracy_evaluation_epochs'] = sorted(set(audit['formal_accuracy_evaluation_epochs'] + audit['diagnostic_accuracy_evaluation_epochs']))
    data.dump(cosmos.MASTER / 'protocol_audit.json', audit)


def sequence(local_pid):
    data.prepare()
    if not (cosmos.MASTER / 'completed.txt').exists():
        status('waiting_for_cosmosqa_local10', existing_child_pid=local_pid)
        wait_process(local_pid, 'run_cosmosqa_sanity.py')
    if not (cosmos.MASTER / 'completed.txt').exists():
        current = data.read(cosmos.MASTER / 'status.json')
        if current['stage'] == 'failed': raise RuntimeError('CosmosQA Local failed; refusing downstream training')
        raise RuntimeError('CosmosQA Local10 completion missing; refusing a duplicate or replacement Local run')
    assert data.read(cosmos.MASTER / 'status.json')['stage'] == 'complete'
    assert data.read(cosmos.MASTER / 'protocol_audit.json')['passed']
    patch_cosmos_diagnostic_report()
    audit_path = data.MASTER / 'protocol_audit.json'
    if not audit_path.exists() or not data.read(audit_path).get('passed'):
        child('smoke', ['scripts/run_cosmosqa_five_experiments.py', '--job', 'smoke'])
    for method in ('fedavg', 'centralized'):
        if not (data.MASTER / method / 'completed.txt').exists():
            child(method, ['scripts/run_cosmosqa_five_experiments.py', '--job', method])
    child('final_report', ['scripts/run_cosmosqa_five_experiments.py', '--job', 'report'])
    status('complete', report=str(ROOT / 'reports/cosmosqa_five_comparison_seed42.md'))
    (QUEUE / 'completed.txt').write_text('Local10 -> FedAvg10 -> Centralized10 -> report complete.\n', encoding='utf-8')


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'): stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-pid', type=int, required=True)
    args = parser.parse_args()
    QUEUE.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    with (QUEUE / 'queue.lock').open('a+b') as lock:
        if lock.tell() == 0: lock.write(b'0'); lock.flush()
        lock.seek(0)
        try: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError: raise SystemExit('CosmosQA five-domain sequence already running')
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            sequence(args.local_pid)
        except BaseException as error:
            old = data.read(QUEUE / 'status.json') if (QUEUE / 'status.json').exists() else {}
            status('failed', failed_stage=old.get('stage'), error=str(error), traceback=traceback.format_exc())
            traceback.print_exc(); raise SystemExit(1)
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__': main()
