#!/usr/bin/env python3
"""Pinned public DrivingWorld weights (Hugging Face, exact revision), only with the owner's approval record.

  download_drivingworld_weights.py --dest DIR --print                       # plan only, no network
  download_drivingworld_weights.py --dest DIR --approval APPROVAL.json      # download + verify

The approval record (private, outside git) must say owner_approved: true, approved_by: "owner", license: "MIT" and name
the exact repo and revision. It is checked before any mkdir or network request.
Every file is fetched from .../resolve/<revision>/<file> into a .partial file, verified (size and sha256, or the git blob
sha1 for small files), then hard-linked into place; an existing mismatched file or a leftover partial is never replaced.
Writes SHA256SUMS, PROVENANCE.json and download-verified.json into DIR.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

sys.dont_write_bytecode = True
import drivingworld_common as dc  # noqa: E402

W = dc.PINS['weights']


def approval_ok(path):
    if path is None or not Path(path).is_file():
        raise ValueError('owner approval record required; no download started')
    record = json.loads(Path(path).read_text())
    if (record.get('owner_approved') is not True or record.get('approved_by') != 'owner'
            or record.get('repo') != W['repo'] or record.get('revision') != W['revision']
            or record.get('license') != 'MIT'):
        raise ValueError('owner approval scope mismatch')
    return record


def verify(path, row):
    if not path.is_file() or path.stat().st_size != row['bytes']:
        return False
    return dc.sha(path) == row['sha256'] if row.get('sha256') else dc.blob(path) == row['git_blob_sha1']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dest', type=Path, required=True)
    ap.add_argument('--approval', type=Path)
    ap.add_argument('--print', dest='print_only', action='store_true')
    a = ap.parse_args()
    if a.print_only:
        print(json.dumps(dict(repo=W['repo'], revision=W['revision'], files=[r['path'] for r in W['files']],
                              bytes=sum(r['bytes'] for r in W['files']), download_started=False, owner_approval_required=True)))
        return
    record = approval_ok(a.approval)  # before mkdir or any network request
    a.dest.mkdir(parents=True, exist_ok=True)
    log = []
    for row in W['files']:
        target = dc.safe_relative(a.dest, row['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        if verify(target, row):
            log.append(dict(path=row['path'], status='present-verified'))
            continue
        if target.exists():
            raise FileExistsError('existing mismatched file retained: ' + str(target))
        partial = target.with_name(target.name + '.partial')
        if partial.exists():
            raise FileExistsError('partial retained; inspect before retry: ' + str(partial))
        url = 'https://huggingface.co/' + W['repo'] + '/resolve/' + W['revision'] + '/' + row['repo_path']
        request = urllib.request.Request(url, headers={'User-Agent': 'wm-calib-bench-pinned-download/1'})
        started = time.time()
        with urllib.request.urlopen(request, timeout=120) as source, partial.open('xb') as out:
            while True:
                block = source.read(8 << 20)
                if not block:
                    break
                out.write(block)
        if not verify(partial, row):
            raise ValueError('download hash/size mismatch; partial retained: ' + row['path'])
        os.link(partial, target)
        partial.unlink()
        log.append(dict(path=row['path'], status='downloaded-verified', seconds=round(time.time() - started, 1)))
    dc.verify_weights(a.dest, full=True)
    sums = {r['path']: dc.sha(dc.safe_relative(a.dest, r['path'])) for r in W['files']}
    for key in W['pickle_globals_allowed']:
        dc.check_pickle_globals(a.dest / key, key)
    with (a.dest / 'SHA256SUMS').open('x') as f:
        for path, digest in sums.items():
            f.write(f'{digest}  {path}\n')
    dc.new_json(a.dest / 'PROVENANCE.json', dict(repo=W['repo'], revision=W['revision'], license=W['license_name'],
                                                 files=W['files'], sha256=sums, log=log,
                                                 pickle_globals={k: dc.pickle_globals(a.dest / k) for k in W['pickle_globals_allowed']},
                                                 downloader_sha256=dc.sha(Path(__file__)),
                                                 downloaded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    dc.new_json(a.dest / 'download-verified.json', dict(complete=True, repo=W['repo'], revision=W['revision'],
                                                        approval_sha256=dc.sha(a.approval), approval_date=record.get('date'),
                                                        files=len(W['files'])))
    print(json.dumps(dict(complete=True, files=len(W['files']), log=log)))


if __name__ == '__main__':
    main()
