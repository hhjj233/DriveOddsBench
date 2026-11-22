#!/usr/bin/env python3
"""Fresh exact-commit checkout of the released DrivingWorld code (no weights, nothing modified).

  prepare_source.py --dest NEW_DIR

Fetches YvanYin/DrivingWorld at the pinned commit and checks every tracked file by its git blob sha1 (and that no
unregistered Python file sits in the imported packages). The kit imports this tree read-only; it is never patched.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import drivingworld_common as dc  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dest', type=Path, required=True)
    a = ap.parse_args()
    a.dest.mkdir(parents=True, exist_ok=False)

    def git(*args):
        subprocess.run(['git', '-C', str(a.dest), *args], check=True)
    git('init', '--quiet')
    git('remote', 'add', 'origin', dc.PINS['code']['repo'] + '.git')
    git('fetch', '--quiet', '--depth=1', 'origin', dc.PINS['code']['commit'])
    git('checkout', '--quiet', '--detach', 'FETCH_HEAD')
    head = subprocess.check_output(['git', '-C', str(a.dest), 'rev-parse', 'HEAD'], text=True).strip()
    if head != dc.PINS['code']['commit']:
        raise ValueError('checked-out commit differs from the pin')
    print(json.dumps(dict(dest=str(a.dest), **dc.verify_source(a.dest))))


if __name__ == '__main__':
    main()
