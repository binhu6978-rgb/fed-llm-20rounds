"""All four smoke tests, then four serial default baseline runs and final report."""
import ctypes
import msvcrt
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import fedlora_baseline_data as data
from scripts import run_fedlora_baseline as runner
shared=runner.shared
QUEUE=data.MASTER/'sequence'; LOGS=ROOT/'logs/fedlora_baselines_5source_seed42'


def status(stage,**details):
    data.dump(QUEUE/'status.json',dict(stage=stage,pid=os.getpid(),updated_at=shared.now(),**details))


def child(method,smoke):
    import psutil
    stage=('smoke_' if smoke else 'formal_')+method
    old=data.read(QUEUE/'status.json') if (QUEUE/'status.json').exists() else {}
    if old.get('stage')==stage and old.get('child_pid') and psutil.pid_exists(old['child_pid']):
        try:
            p=psutil.Process(old['child_pid'])
            assert any('run_fedlora_baseline.py' in x for x in p.cmdline()) and Path(p.cwd()).resolve()==ROOT
            p.wait()
        except psutil.NoSuchProcess: pass
    arguments=['scripts/run_fedlora_baseline.py','--alg',method,'--suffix',data.suffix(method)]
    for name,value in data.config()['algorithms'][method].items(): arguments += ['--'+name,str(value)]
    if smoke: arguments += ['--smoke']
    path=LOGS/(stage+'.log')
    with path.open('a',encoding='utf-8') as stream:
        stream.write('\nRun '+shared.now()+'\n'); stream.flush()
        p=subprocess.Popen([sys.executable,'-X','utf8','-u',*arguments],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
            env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8'))
        status(stage,child_pid=p.pid,log=str(path))
        code=p.wait()
    if code: raise RuntimeError(stage+' failed; later training halted. '+str(path))


def sequence():
    data.prepare(); runner.report()
    for method in data.METHODS:
        folder=data.output(method)/'smoke'
        if not (folder/'completed.txt').exists(): child(method,True)
        assert data.read(folder/'protocol_audit.json')['passed']
    for method in data.METHODS:
        if not (data.output(method)/'completed.txt').exists(): child(method,False)
        runner.report()
    path,complete=runner.report(); assert complete
    status('complete',report=str(path))
    (QUEUE/'completed.txt').write_text('Four smoke tests and four default baseline runs complete.\n',encoding='utf-8')


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'): stream.reconfigure(encoding='utf-8')
    QUEUE.mkdir(parents=True,exist_ok=True); LOGS.mkdir(parents=True,exist_ok=True)
    with (QUEUE/'queue.lock').open('a+b') as lock:
        if lock.tell()==0: lock.write(b'0'); lock.flush()
        lock.seek(0)
        try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError: raise SystemExit('Baseline queue already running')
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            sequence()
        except BaseException as error:
            old=data.read(QUEUE/'status.json') if (QUEUE/'status.json').exists() else {}
            status('failed',failed_stage=old.get('stage'),error=str(error),traceback=traceback.format_exc())
            try: runner.report()
            except Exception: pass
            traceback.print_exc(); raise SystemExit(1)
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__': main()
