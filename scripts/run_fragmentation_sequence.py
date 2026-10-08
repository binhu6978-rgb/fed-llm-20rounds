"""Serial Setting B queue: prepare -> smoke -> Local10 -> FedAvg10 -> report."""
import ctypes
import msvcrt
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import fragmentation_data as data
from scripts import run_three_dataset_lora as shared

QUEUE = data.MASTER/'sequence'
LOGS = ROOT/'logs/fragmentation_5source_10client_seed42'


def status(stage,**details):
    data.dump(QUEUE/'status.json',dict(stage=stage,pid=os.getpid(),updated_at=shared.now(),**details))


def child(stage):
    # Resume an orphaned queue child before deciding whether its stage needs rerun.
    import psutil
    old=data.read(QUEUE/'status.json') if (QUEUE/'status.json').exists() else {}
    pid=old.get('child_pid') if old.get('stage')==stage else None
    if pid and psutil.pid_exists(pid):
        try:
            process=psutil.Process(pid)
            assert any('run_fragmentation_experiment.py' in x for x in process.cmdline())
            assert Path(process.cwd()).resolve()==ROOT
            process.wait()
        except psutil.NoSuchProcess: pass
    path=LOGS/(stage+'.log')
    with path.open('a',encoding='utf-8') as stream:
        stream.write('\nRun '+shared.now()+'\n'); stream.flush()
        process=subprocess.Popen([sys.executable,'-X','utf8','-u','scripts/run_fragmentation_experiment.py','--job',stage],
            cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
            env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8'))
        status(stage,child_pid=process.pid,log=str(path))
        code=process.wait()
    if code: raise RuntimeError(f'{stage} failed (exit {code}); later stages stopped. See {path}')


def sequence():
    data.prepare()
    runner=ROOT/'scripts/run_fragmentation_experiment.py'
    smoke=data.MASTER/'smoke_audit.json'
    ready=smoke.exists() and data.read(smoke).get('passed')
    if ready:
        protocol=data.read(smoke)['protocol']
        ready=protocol['suite_runner_sha256']==data.sha(runner) and protocol['data_helper_sha256']==data.sha(Path(data.__file__))
    if not ready: child('smoke')
    for stage,method in (('local','clientlocal'),('fedavg','fedavg')):
        if not (data.MASTER/method/'completed.txt').exists(): child(stage)
    child('report')
    status('complete',report=str(ROOT/'reports/fragmentation_5source_10client_seed42.md'))
    (QUEUE/'completed.txt').write_text('Setting B Local -> FedAvg -> report complete.\n',encoding='utf-8')


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'): stream.reconfigure(encoding='utf-8')
    QUEUE.mkdir(parents=True,exist_ok=True); LOGS.mkdir(parents=True,exist_ok=True)
    with (QUEUE/'queue.lock').open('a+b') as lock:
        if lock.tell()==0: lock.write(b'0'); lock.flush()
        lock.seek(0)
        try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError: raise SystemExit('Fragmentation sequence already running')
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            sequence()
        except BaseException as error:
            old=data.read(QUEUE/'status.json') if (QUEUE/'status.json').exists() else {}
            status('failed',failed_stage=old.get('stage'),error=str(error),traceback=traceback.format_exc())
            traceback.print_exc(); raise SystemExit(1)
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__': main()
