"""Run FFA-LoRA then FedSVD, each fresh rounds1-20; resume with the same command."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_ffalora20_server import (
    MASTER, METHODS, NAMES, configure_environment, dump, file_lock, now,
    preflight, read, relative, status,
)


def combined_report():
    summaries = {}
    lines = ['# FFA-LoRA / FedSVD: fresh 20-round experiments', '',
        'Both start from the canonical seed42 adapter; no round10 states reused.', '',
        '| Method | Completed | Best round | Peak shared Macro (%) | Round20 Macro (%) |',
        '|---|---:|---:|---:|---:|']
    for method in METHODS:
        path = MASTER / method / 'result.json'
        if not path.exists():
            lines.append('| ' + NAMES[method] + ' | pending | — | — | — |')
            continue
        r = read(path)
        summaries[method] = r
        final = '—' if r['round20_macro'] is None else '%.2f' % (r['round20_macro'] * 100)
        lines.append('| %s | %d/20 | %d | %.2f | %s |' % (
            NAMES[method], r['completed_round'], r['best_round'], r['peak_shared_macro'] * 100, final))
    dump(MASTER / 'result.json', summaries)
    (MASTER / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Read input counts/IDs only; no training')
    parser.add_argument('--gpu', help='Optional CUDA_VISIBLE_DEVICES; preserves scheduler environment by default')
    args = parser.parse_args()
    configure_environment(args.gpu)
    if args.check:
        preflight()
        print('INPUT CHECK PASSED: both methods start at round1; train=20000 test=7498', flush=True)
        return
    with file_lock(MASTER / 'sequence.lock'):
        try:
            status(MASTER / 'sequence_status.json', 'checking_inputs')
            preflight()
            logs = MASTER / 'logs'
            logs.mkdir(exist_ok=True)
            combined_report()
            for method in METHODS:
                log = logs / (method + '.log')
                command = [sys.executable, '-X', 'utf8', '-u', str(ROOT / 'scripts' / f'run_{method}20_server.py')]
                if args.gpu is not None:
                    command += ['--gpu', args.gpu]
                # Each child owns its method lock. Its process exiting also frees GPU memory.
                with log.open('a', encoding='utf-8') as f:
                    f.write('\nSTART ' + now() + '\n'); f.flush()
                    process = subprocess.Popen(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                        env=dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8'))
                    status(MASTER / 'sequence_status.json', 'running', method=method,
                           child_pid=process.pid, log=relative(log))
                    code = process.wait()
                if code:
                    raise RuntimeError(f'{NAMES[method]} failed; remaining methods stopped. See {log}')
                folder = MASTER / method
                assert (folder / 'completed.txt').is_file()
                assert read(folder / 'protocol_audit.json')['passed']
                assert read(folder / 'result.json')['completed_round'] == 20
                combined_report()
            status(MASTER / 'sequence_status.json', 'complete', report=relative(MASTER / 'report.md'))
        except BaseException as error:
            status(MASTER / 'sequence_status.json', 'failed', error=str(error), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
