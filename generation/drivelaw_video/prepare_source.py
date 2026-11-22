#!/usr/bin/env python3
"""Create a fresh exact-revision public inference checkout; no model weights."""
import argparse
from pathlib import Path
import subprocess
import sys
import io
import tarfile
sys.dont_write_bytecode=True
import drivelaw_common as dc

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--dest',type=Path,required=True);a=ap.parse_args()
    a.dest.mkdir(parents=True,exist_ok=False)
    def git(*args,input=None):subprocess.run(['git','-C',str(a.dest),*args],input=input,text=True,check=True)
    git('init','--quiet');git('remote','add','origin',dc.PINS['code']['repo']+'.git')
    git('fetch','--depth=1','--filter=blob:none','origin',dc.PINS['code']['commit'])
    git('sparse-checkout','set','--no-cone','--stdin',input='/DriveLaW-Video/Infer/infer.py\n/DriveLaW-Video/Infer/diffusers/src/diffusers/\n/DriveLaW-Video/Infer/diffusers/setup.py\n/DriveLaW-Video/Infer/diffusers/README.md\n')
    git('checkout','--detach','FETCH_HEAD')
    selected=dc.PINS['code']['reconstructed_models'];up=a.dest/'.bench-upstream-models';up.mkdir(exist_ok=False)
    def upstream(*args):subprocess.run(['git','-C',str(up),*args],check=True)
    upstream('init','--quiet');upstream('remote','add','origin','https://github.com/'+selected['upstream_repo']+'.git')
    upstream('fetch','--depth=1','--filter=blob:none','origin',selected['commit'])
    archive=subprocess.check_output(['git','-C',str(up),'archive',selected['commit'],'src/diffusers/models'])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if member.isfile():
                dest=dc.safe_relative(a.dest/'DriveLaW-Video/Infer/diffusers',member.name)
                dest.parent.mkdir(parents=True,exist_ok=True)
                with dest.open('xb') as f:f.write(tar.extractfile(member).read())
    print(dc.verify_source(a.dest));dc.verify_runtime_complete(a.dest)
if __name__=='__main__':main()
