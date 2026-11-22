#!/usr/bin/env python3
"""Create a fresh exact-commit checkout of the official ReWorld Stage-1 source (no weights, no DriveLaW-Act)."""
import argparse
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
import reworld_common as rc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dest', type=Path, required=True)
    a = ap.parse_args()
    a.dest.mkdir(parents=True, exist_ok=False)

    def git(*args, input=None):
        subprocess.run(['git', '-C', str(a.dest), *args], input=input, text=True, check=True)

    git('init', '--quiet')
    git('remote', 'add', 'origin', rc.PINS['code']['repo'] + '.git')
    git('fetch', '--quiet', '--depth=1', '--filter=blob:none', 'origin', rc.PINS['code']['commit'])
    git('sparse-checkout', 'set', '--no-cone', '--stdin', input='\n'.join(rc.PINS['code']['sparse_paths']) + '\n')
    git('checkout', '--quiet', '--detach', 'FETCH_HEAD')
    head = subprocess.check_output(['git', '-C', str(a.dest), 'rev-parse', 'HEAD'], text=True).strip()
    if head != rc.PINS['code']['commit']:
        raise ValueError('checked-out commit differs from the pin')
    print(rc.verify_source(a.dest))


if __name__ == '__main__':
    main()
