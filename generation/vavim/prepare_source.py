#!/usr/bin/env python3
"""Fresh exact-commit checkout of the released VaViM / VaVAM code (no weights, nothing modified).

  prepare_source.py --dest NEW_DIR

Fetches valeoai/VideoActionModel at the pinned commit (main 738050e, which carries the #57 rolling-context fix on top of
the v1.0.0 release) and checks every tracked file by its git blob sha1 (and that no unregistered Python file sits under
vam/). The kit imports this tree read-only (vam.video_pretraining); it is never patched or pip-installed.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import vavim_common as vc  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dest', type=Path, required=True)
    a = ap.parse_args()
    a.dest.mkdir(parents=True, exist_ok=False)

    def git(*args):
        subprocess.run(['git', '-C', str(a.dest), *args], check=True)
    git('init', '--quiet')
    git('remote', 'add', 'origin', vc.PINS['code']['repo'] + '.git')
    git('fetch', '--quiet', '--depth=1', 'origin', vc.PINS['code']['commit'])
    git('checkout', '--quiet', '--detach', 'FETCH_HEAD')
    head = subprocess.check_output(['git', '-C', str(a.dest), 'rev-parse', 'HEAD'], text=True).strip()
    if head != vc.PINS['code']['commit']:
        raise ValueError('checked-out commit differs from the pin')
    print(json.dumps(dict(dest=str(a.dest), **vc.verify_source(a.dest))))


if __name__ == '__main__':
    main()
