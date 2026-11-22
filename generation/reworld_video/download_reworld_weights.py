#!/usr/bin/env python3
"""Pinned public weights (tz2026/ReWorld Video DiT + LTX-Video 0.9.5 VAE/scheduler), only after owner licence approval.

Downloads exactly the files listed in reworld_pins.json at the pinned revisions, verifies size and SHA256 of every
file, refuses to replace mismatched or partial files, and writes download-verified.json plus SHA256SUMS.
No token is read or sent: both repositories are public and ungated (checked at pin time).
"""
import argparse
import json
import os
from pathlib import Path
import sys
import urllib.request
sys.dont_write_bytecode = True
import reworld_common as rc


def license_ok(path):
    if path is None or not Path(path).is_file():
        raise ValueError('owner licence approval record required; no download started')
    record = json.loads(Path(path).read_text())
    if record.get('owner_approved') is not True or record.get('approved_by') != 'owner':
        raise ValueError('owner approval absent')
    approved = {(r.get('repo'), r.get('revision'), r.get('license_sha256')) for r in record.get('repos', [])}
    for key, repo in rc.PINS['weights']['repos'].items():
        lic = next(x for x in rc.PINS['weights']['files'] if x['repo_key'] == key and x['repo_path'] == repo['license_path'])
        if (repo['repo'], repo['revision'], lic['sha256']) not in approved:
            raise ValueError('owner licence approval scope/hash mismatch: ' + repo['repo'])
    return record


def verify(path, row):
    return path.is_file() and path.stat().st_size == row['bytes'] and rc.sha(path) == row['sha256']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dest', type=Path, required=True)
    ap.add_argument('--license-approval', type=Path)
    ap.add_argument('--print', dest='print_only', action='store_true')
    a = ap.parse_args()
    files = rc.PINS['weights']['files']
    if a.print_only:
        print(json.dumps(dict(repos={k: [v['repo'], v['revision']] for k, v in rc.PINS['weights']['repos'].items()},
                              files=len(files), bytes=sum(x['bytes'] for x in files), download_started=False,
                              owner_license_required=True)))
        return
    record = license_ok(a.license_approval)  # before mkdir or any network request
    a.dest.mkdir(parents=True, exist_ok=True)
    for row in files:
        repo = rc.PINS['weights']['repos'][row['repo_key']]
        target = rc.safe_relative(a.dest, row['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        if verify(target, row):
            continue
        if target.exists():
            raise FileExistsError('existing mismatched file retained: ' + str(target))
        partial = target.with_name(target.name + '.partial')
        if partial.exists():
            raise FileExistsError('partial retained; inspect before retry: ' + str(partial))
        url = 'https://huggingface.co/' + repo['repo'] + '/resolve/' + repo['revision'] + '/' + row['repo_path']
        request = urllib.request.Request(url, headers={'User-Agent': 'wm-calib-bench-pinned-download/1'})
        with urllib.request.urlopen(request, timeout=120) as source, partial.open('xb') as dest:
            while True:
                block = source.read(4 << 20)
                if not block:
                    break
                dest.write(block)
        if not verify(partial, row):
            raise ValueError('download hash/size mismatch; partial retained: ' + str(partial))
        os.link(partial, target)
        partial.unlink()
        print(json.dumps(dict(verified=row['path'], bytes=row['bytes'])), flush=True)
    rc.verify_weights(a.dest, full=True)
    sums = ''.join(f"{row['sha256']}  {row['path']}\n" for row in files)
    for name, text in (('SHA256SUMS', sums),):
        p = a.dest / name
        if p.exists() and p.read_text() != text:
            raise FileExistsError('existing ' + name + ' differs')
        if not p.exists():
            p.write_text(text)
    rc.new_json(a.dest / 'download-verified.json',
                dict(complete=True, repos={k: [v['repo'], v['revision']] for k, v in rc.PINS['weights']['repos'].items()},
                     license_approval_sha256=rc.sha(a.license_approval), approved_at=record.get('approved_at'),
                     files=len(files), bytes=sum(x['bytes'] for x in files), pins_sha256=rc.sha(rc.PINS_PATH)))
    print(json.dumps(dict(complete=True, files=len(files))))


if __name__ == '__main__':
    main()
