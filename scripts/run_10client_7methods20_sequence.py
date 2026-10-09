"""Serial fresh 10-client/7-method/20-round experiments; rerun to resume."""
import argparse
import csv
import errno
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_10client_7methods20_server import (
    DOMAINS, MASTER, METHODS, NAMES, configure_environment, dump, file_lock,
    is_complete, now, preflight, read, relative, status,
)


def combined_report():
    summaries, flat = {}, []
    lines = ['# Fresh 10-client / 7-method / 20-round experiments', '',
        'Existing seed42 split; all methods start at the same canonical initial LoRA.',
        'Main result: best unrounded five-domain shared Macro; earliest tie. All task scores come from that same checkpoint.', '',
        '| Method | Completed | Best round | Best shared Macro (%) | Round20 Macro (%) |',
        '|---|---:|---:|---:|---:|---:|']
    for method in METHODS:
        path = MASTER / method / 'result.json'
        if not path.exists():
            lines.append('| ' + NAMES[method] + ' | pending | — | — | — |')
            continue
        result = read(path)
        summaries[method] = result
        final = '—' if result['round20_macro'] is None else '%.2f' % (result['round20_macro'] * 100)
        lines.append('| %s | %d/20 | %d | %.2f | %s |' % (NAMES[method], result['completed_round'],
            result['best_round'], result['best_shared_macro'] * 100, final))
        flat.append(dict(method=method, completed_round=result['completed_round'], best_round=result['best_round'],
            **{d + '_accuracy': result['best_metrics'][d + '_accuracy'] for d in DOMAINS},
            best_shared_macro=result['best_shared_macro'], round20_macro=result['round20_macro'],
            best_checkpoint=result['best_checkpoint'], best_predictions=result['best_predictions']))
    dump(MASTER / 'result.json', summaries)
    if flat:
        path = MASTER / 'best_shared_results.csv'
        temporary = path.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0]))
            writer.writeheader(); writer.writerows(flat)
        os.replace(temporary, path)
    (MASTER / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def child(method, gpu):
    folder = MASTER / method
    # A worker can survive an interrupted queue. Wait for its lock instead of
    # allocating a duplicate model or terminating that existing worker.
    while True:
        try:
            with file_lock(folder / 'run.lock'):
                pass
            break
        except OSError as error:
            if error.errno not in (errno.EAGAIN, errno.EACCES, errno.EDEADLK):
                raise
            status(MASTER / 'sequence_status.json', 'waiting_existing_worker', method=method)
            time.sleep(10)
    if is_complete(method):
        print(NAMES[method], 'already completed20; skipped', flush=True)
        return
    logdir = MASTER / 'logs'
    logdir.mkdir(exist_ok=True)
    log = logdir / (method + '.log')
    command = [sys.executable, '-X', 'utf8', '-u', str(ROOT / 'scripts/run_10client_7methods20_server.py'), '--alg', method]
    if gpu is not None:
        command += ['--gpu', gpu]
    with log.open('a', encoding='utf-8') as f:
        f.write('\nSTART ' + now() + '\n'); f.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8'))
        status(MASTER / 'sequence_status.json', 'running', method=method,
               child_pid=process.pid, log=relative(log))
        code = process.wait()
    if code:
        raise RuntimeError(f'{NAMES[method]} failed; remaining methods stopped. See {log}')
    assert is_complete(method), f'{method} exited without complete20 results'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Read existing split and canonical input counts/IDs only; no training')
    parser.add_argument('--gpu', help='Optional CUDA_VISIBLE_DEVICES; preserve scheduler environment by default')
    args = parser.parse_args()
    configure_environment(args.gpu)
    if args.check:
        preflight()
        print('INPUT CHECK PASSED: existing 10-client split; seven methods fresh rounds1-20', flush=True)
        return
    with file_lock(MASTER / 'sequence.lock'):
        try:
            status(MASTER / 'sequence_status.json', 'checking_inputs')
            preflight()
            combined_report()
            for method in METHODS:
                child(method, args.gpu)
                combined_report()
            status(MASTER / 'sequence_status.json', 'complete', report=relative(MASTER / 'report.md'))
        except BaseException as error:
            status(MASTER / 'sequence_status.json', 'failed', error=str(error), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
