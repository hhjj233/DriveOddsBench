#!/usr/bin/env python3
"""DEV-only smoke: one registered spec per bounded, owned GPU-wrapper session."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.dont_write_bytecode=True
import drivelaw_common as dc
from owned_session import OwnedSession,session_rows
from run_drivelaw_bench import validate_spec

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['science-python','bench','input-root','output-root','src','weights','wrapper','ledger']:
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--wrapper-sha256',required=True);ap.add_argument('--specs',type=Path,nargs='+',required=True)
    ap.add_argument('--print',dest='print_only',action='store_true');ap.add_argument('--run',action='store_true')
    ap.add_argument('--need-mib',type=int,default=20000);ap.add_argument('--min-host-gib',type=int,default=36)
    a=ap.parse_args();jobs=[]
    if dc.sha(a.wrapper)!=a.wrapper_sha256:raise ValueError('approved unchanged wrapper hash required')
    dc.verify_source(a.src)
    for path in a.specs:
        spec=json.loads(path.read_text())
        if spec.get('split')!='dev':raise ValueError('local smoke permits DEV only')
        validate_spec(spec,a.bench,a.input_root)
        dest=dc.safe_relative(a.output_root,spec['output_dir'])
        if dest.exists():raise FileExistsError('fresh outputs required; no retry')
        cmd=['bash',str(a.wrapper),'timeout','--foreground','--signal=TERM','--kill-after=60s','1800s',str(a.science_python),'-B',str(dc.HERE/'run_drivelaw_bench.py'),
             '--specs',str(path),'--bench',str(a.bench),'--input-root',str(a.input_root),'--output-root',str(a.output_root),'--src',str(a.src),'--weights',str(a.weights)]
        jobs.append(dict(name=spec['name'],command=cmd,spec_sha256=dc.sha(path),expected_output=str(dest/'sample_outputs.json')))
    if a.ledger.exists():raise FileExistsError('fresh ledger required')
    if a.print_only:
        print(json.dumps(dict(CPU_only=True,GPU_started=False,jobs=jobs,seconds_per_job=1800,wrapper_wait_minutes=30,NEED_MIB=a.need_mib,owner_license_required=True,reference_policy_approved=dc.B['reference_policy_approved']),indent=2));return
    if not a.run:raise ValueError('--print or explicit --run required')
    dc.verify_runtime_complete(a.src)
    if not dc.B['reference_policy_approved']:raise ValueError('codec review required before GPU')
    proof=json.loads((a.weights/'download-verified.json').read_text())
    if not proof.get('complete') or proof.get('revision')!=dc.PINS['weights']['revision']:raise ValueError('approved pinned download proof required')
    dc.verify_weights(a.weights)
    # Supervision needs the host interpreter with working pidfd APIs.
    fd=os.pidfd_open(os.getpid(),0);os.close(fd)
    if not hasattr(signal,'pidfd_send_signal'):raise ValueError('pidfd signal API unavailable')
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))/2**20
    if available<a.min_host_gib:raise ValueError('host MemAvailable below requested margin')
    stop={'signal':None}
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(sig,lambda n,f:stop.update(signal=n))
    a.ledger.mkdir(parents=True,exist_ok=False)
    a.output_root.mkdir(parents=True,exist_ok=True)
    with (a.output_root/'drivelaw-controller.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        dc.new_json(a.ledger/'plan.json',dict(jobs=jobs,pins_sha256=dc.sha(dc.PINS_PATH),wrapper_sha256=a.wrapper_sha256))
        for job in jobs:
            if stop['signal']:raise RuntimeError('stop before next job')
            sd=a.ledger/job['name'];sd.mkdir();owned=None;start=time.monotonic()
            try:
                env=dict(os.environ,NEED_MIB=str(a.need_mib),WAIT_MIN='30',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
                with (sd/'job.log').open('x') as log:
                    child=subprocess.Popen(job['command'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    owned=OwnedSession(child,os.getpid())
                    while child.poll() is None and not owned.handles:
                        owned.bind();time.sleep(.02)
                    dc.new_json(sd/'pids.json',dict(supervisor=os.getpid(),own_pgid=child.pid,birth=owned.birth,wrapper_sha256=a.wrapper_sha256))
                    while child.poll() is None:
                        if stop['signal'] or time.monotonic()-start>3660:raise RuntimeError('controlled stop/timeout')
                        time.sleep(.2)
                    child.wait()
                if stop['signal'] or child.returncode or session_rows(child.pid):raise RuntimeError('nonzero/cancelled/or surviving own children; no retry')
                result=json.loads(Path(job['expected_output']).read_text())
                if result.get('status')!='success' or result.get('spec_sha256')!=job['spec_sha256']:raise ValueError('sample output provenance mismatch')
                dc.new_json(sd/'job.done',dict(returncode=0,seconds=time.monotonic()-start,ownPGID_live=[]))
            except BaseException as exc:
                if owned:
                    while True:
                        try:
                            if owned.cleanup():break
                        except (OSError,RuntimeError):pass
                        print('Owned cleanup pending; controller slot retained',flush=True);time.sleep(.2)
                dc.new_json(sd/'job.failed',dict(error_type=type(exc).__name__,no_retry=True,all_outputs_retained=True))
                raise
            finally:
                if owned:owned.close()
            print(json.dumps(dict(job=job['name'],status='done')),flush=True)
if __name__=='__main__':main()
