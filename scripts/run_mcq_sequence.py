"""Wait for existing Skewed FedAvg, then Centralized-All, then FedAvg-IID.

Each child must finish successfully before the next starts. Resumes use existing
client/epoch checkpoints. No simultaneous GPU jobs and no polling by the user.
"""
import argparse
import ctypes
import json
import msvcrt
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_mcq_dataset_skewed import dump, read

QUEUE = ROOT / 'exp/mcq_controls_seed42'
LOGS = ROOT / 'logs/mcq_controls_seed42'


def status(stage, **detail):
    dump(QUEUE / 'status.json', dict(stage=stage, pid=os.getpid(),
         updated_at=datetime.now(timezone.utc).isoformat(), **detail))


def child(stage, command, log_path):
    print(f'Start {stage}: {log_path}', flush=True)
    with log_path.open('a', encoding='utf-8') as log:
        log.write(f'\n=== {stage} {datetime.now(timezone.utc).isoformat()} ===\n')
        log.flush()
        environment = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment)
        status(stage, child_pid=process.pid, log=str(log_path))
        result = process.wait()
    if result != 0:
        raise RuntimeError(f'{stage} failed with exit code {result}; see {log_path}')
    print(f'Finished {stage}', flush=True)


def wait_for_skewed(pid):
    complete = ROOT / 'exp/mcq_dataset_skewed_seed42/completed.txt'
    process = None
    if pid is None:
        # A rerun of the queue must attach to a surviving Skewed child rather
        # than start another copy of the GPU job.
        for candidate in psutil.process_iter(['pid', 'cmdline']):
            try:
                command = candidate.info['cmdline'] or []
                if (any('run_mcq_dataset_skewed.py' in x for x in command)
                        and ('--job' not in command or command[command.index('--job') + 1] == 'run')
                        and Path(candidate.cwd()).resolve() == ROOT):
                    if pid is not None:
                        raise RuntimeError('Multiple Skewed runners found; refusing to add another GPU job')
                    pid = candidate.pid
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    if pid is not None and psutil.pid_exists(pid):
        process = psutil.Process(pid)
        command = process.cmdline()
        assert any('run_mcq_dataset_skewed.py' in x for x in command), 'PID is not the expected Skewed runner'
        assert Path(process.cwd()).resolve() == ROOT, 'PID belongs to a different workspace'
        created = process.create_time()
        last = None
        while process.is_running() and process.create_time() == created:
            path = ROOT / 'exp/mcq_dataset_skewed_seed42/live_progress.json'
            progress = read(path) if path.exists() else {}
            if progress != last:
                status('waiting_for_dataset_skewed', existing_child_pid=pid, progress=progress)
                print('Waiting for Skewed: ' + json.dumps(progress), flush=True)
                last = progress
            time.sleep(10)
    if not complete.exists():
        # A prior interrupted process can be resumed, but a failed recovery
        # stops this sequence rather than consuming GPU on the later controls.
        child('dataset_skewed_resume', [sys.executable, '-u', str(ROOT / 'scripts/run_mcq_dataset_skewed.py'),
              '--job', 'run'], LOGS / 'dataset_skewed_resume.log')
    if not complete.exists():
        raise RuntimeError('Skewed completion marker missing; controls were not started')
    child('dataset_skewed_report', [sys.executable, str(ROOT / 'scripts/run_mcq_dataset_skewed.py'),
          '--job', 'report'], LOGS / 'dataset_skewed_report.log')


def sequence(pid):
    wait_for_skewed(pid)
    for method in ('centralized', 'iid'):
        # If a previous queue was interrupted, its training child may still be
        # alive. Attach to it before resuming, avoiding a second GPU process.
        matches = []
        for candidate in psutil.process_iter(['pid', 'cmdline']):
            try:
                command = candidate.info['cmdline'] or []
                if (any('run_mcq_controls.py' in x for x in command)
                        and '--job' in command and command[command.index('--job') + 1] == 'run'
                        and '--method' in command and command[command.index('--method') + 1] == method
                        and Path(candidate.cwd()).resolve() == ROOT):
                    matches.append(candidate)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if len(matches) > 1:
            raise RuntimeError(f'Multiple {method} runners found; refusing another GPU job')
        if matches:
            process = matches[0]
            created = process.create_time()
            status('waiting_for_' + method, existing_child_pid=process.pid)
            while process.is_running() and process.create_time() == created:
                time.sleep(10)
        child(method, [sys.executable, '-u', str(ROOT / 'scripts/run_mcq_controls.py'),
              '--job', 'run', '--method', method], LOGS / f'{method}.log')
        if not (ROOT / f'exp/mcq_{method}_seed42/completed.txt').exists():
            raise RuntimeError(f'{method}: completion marker missing')
    child('final_report', [sys.executable, str(ROOT / 'scripts/run_mcq_controls.py'),
          '--job', 'suite-report'], LOGS / 'final_report.log')
    status('complete', report=str(ROOT / 'reports/mcq_comparison_seed42.md'))
    (QUEUE / 'completed.txt').write_text('Skewed, Centralized-All, FedAvg-IID and final report completed.\n', encoding='utf-8')
    print('All experiments and comparison report completed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skewed-pid', type=int,
                        help='PID of this workspace\'s already running Skewed job')
    args = parser.parse_args()
    os.chdir(ROOT)
    QUEUE.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with (QUEUE / 'queue.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit('A sequence is already running; no duplicate job was started.')
        try:
            # Keep unattended GPU work running while allowing the display to
            # sleep. This per-thread request disappears on process exit and
            # does not change the user's persistent Windows power plan.
            awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            if not awake:
                raise ctypes.WinError()
            print('Temporary system-awake request active; display sleep is allowed.', flush=True)
            sequence(args.skewed_pid)
        except BaseException as error:
            previous = read(QUEUE / 'status.json').get('stage') if (QUEUE / 'status.json').exists() else 'starting'
            status('failed', failed_stage=previous, error=str(error), logs=str(LOGS))
            traceback.print_exc()
            raise SystemExit(1)
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
