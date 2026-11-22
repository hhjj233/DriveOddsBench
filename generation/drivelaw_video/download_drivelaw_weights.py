#!/usr/bin/env python3
"""Pinned public weights; requires a licence approval record."""
import argparse
import json
from pathlib import Path
import urllib.request
import sys
sys.dont_write_bytecode=True
import drivelaw_common as dc

def license_ok(path):
    if path is None or not Path(path).is_file():raise ValueError('owner Open RAIL-M approval record required; no download started')
    record=json.loads(Path(path).read_text())
    expected=next(x['sha256'] for x in dc.PINS['weights']['files'] if x['path']==dc.PINS['weights']['license_path'])
    if (record.get('owner_approved') is not True or record.get('approved_by')!='owner'
        or record.get('repo')!=dc.PINS['weights']['repo'] or record.get('revision')!=dc.PINS['weights']['revision']
        or record.get('license_sha256')!=expected):raise ValueError('owner licence approval scope/hash mismatch')
    return record

def verify(path,row):
    if not path.is_file() or path.stat().st_size!=row['bytes']:return False
    return dc.sha(path)==row['sha256'] if row.get('sha256') else dc.blob(path)==row['git_blob_sha1']

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--dest',type=Path,required=True);ap.add_argument('--license-approval',type=Path);ap.add_argument('--print',dest='print_only',action='store_true');a=ap.parse_args()
    if a.print_only:
        print(json.dumps(dict(repo=dc.PINS['weights']['repo'],revision=dc.PINS['weights']['revision'],files=len(dc.PINS['weights']['files']),bytes=sum(x['bytes'] for x in dc.PINS['weights']['files']),download_started=False,owner_license_required=True)));return
    license_ok(a.license_approval)  # Before mkdir or any network request.
    a.dest.mkdir(parents=True,exist_ok=True)
    for row in dc.PINS['weights']['files']:
        target=dc.safe_relative(a.dest,row['path']);target.parent.mkdir(parents=True,exist_ok=True)
        if verify(target,row):continue
        if target.exists():raise FileExistsError('existing mismatched file retained: '+str(target))
        partial=target.with_name(target.name+'.partial')
        if partial.exists():raise FileExistsError('partial retained; inspect before retry: '+str(partial))
        url='https://huggingface.co/'+dc.PINS['weights']['repo']+'/resolve/'+dc.PINS['weights']['revision']+'/'+row['repo_path']
        request=urllib.request.Request(url,headers={'User-Agent':'wm-calib-bench-pinned-download/1'})
        with urllib.request.urlopen(request,timeout=120) as source,partial.open('xb') as dest:
            while True:
                block=source.read(4<<20)
                if not block:break
                dest.write(block)
        if not verify(partial,row):raise ValueError('download hash/size mismatch; partial retained')
        import os
        os.link(partial,target);partial.unlink()
    dc.verify_weights(a.dest,full=True)
    dc.new_json(a.dest/'download-verified.json',dict(complete=True,repo=dc.PINS['weights']['repo'],revision=dc.PINS['weights']['revision'],license_approval_sha256=dc.sha(a.license_approval),files=len(dc.PINS['weights']['files'])))
    print(json.dumps(dict(complete=True,files=len(dc.PINS['weights']['files']))))
if __name__=='__main__':main()
