#!/usr/bin/env python3
"""Download of the pinned VaViM-L weights and the LlamaGen tokenizer/detokenizer (GitHub release v1.0.0).

  download_vavim_weights.py --approval OWNER_APPROVAL_JSON --dest NEW_DIR
  download_vavim_weights.py --approval OWNER_APPROVAL_JSON --dest DIR --verify-only

Refuses before any network request or directory creation unless the owner approval record exists and covers exactly
VaViM-L and its tokenizer. Downloads only the five release assets pinned in vavim_pins.json (weights.assets: the three
parts of width_2048_pretrained_139k_total_155k_chunked.tar.gz and the two TorchScript files), each streamed to
<name>.partial, checked for size and sha256, then renamed (a failed partial is kept for diagnosis). The three parts are
then concatenated in memory-bounded streaming into the release's tar.gz (official scripts/handle_checkpoints.py
`cat ... | tar xzf`), whose single regular member width_2048_pretrained_139k_total_155k.pt is extracted (no paths, no
links; gzip CRC and tar size checked by the reader) and checked for size and sha256. The checkpoint's pickle is scanned
by opcode (nothing executed) against the allow-list. Writes SHA256SUMS, PROVENANCE.json and download-verified.json.
Never accepts any licence or gate: the approval record is the owner's.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import urllib.request

sys.dont_write_bytecode = True
import vavim_common as vc  # noqa: E402

CHUNK = 8 << 20


def check_approval(path):
    """The owner's record must exist and name VaViM-L and its tokenizer from the official release (checked offline)."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit('refusing: owner approval record missing (no download, nothing created)')
    record = json.loads(path.read_text())
    scope = ' '.join(str(record.get(k, '')) for k in ('model', 'scope'))
    need = ('VaViM', 'VaViM-L', 'tokenizer', 'v1.0.0')
    if any(word not in scope for word in need) or not record.get('approved_by') or not record.get('licence'):
        raise SystemExit('refusing: approval record does not cover VaViM-L + tokenizer from release v1.0.0')
    return dict(path=str(path), sha256=vc.sha(path), approved_by=record.get('approved_by'),
                approved_utc=record.get('approved_utc'), scope=record.get('scope'), not_included=record.get('not_included'))


def fetch(url, dest, size, digest):
    """Stream one asset to dest.partial with a running sha256; rename only after size and hash match."""
    partial = dest.with_name(dest.name + '.partial')
    if dest.exists() or partial.exists():
        raise FileExistsError(f'{dest} or its partial exists; no overwrite')
    h, n = hashlib.sha256(), 0
    req = urllib.request.Request(url, headers={'User-Agent': 'wm-calib-bench-vavim-kit'})
    with urllib.request.urlopen(req, timeout=120) as r, partial.open('xb') as f:
        while True:
            block = r.read(CHUNK)
            if not block:
                break
            f.write(block)
            h.update(block)
            n += len(block)
    if n != size or h.hexdigest() != digest:
        raise ValueError(f'{dest.name}: size {n} sha256 {h.hexdigest()} differs from the pins (partial kept)')
    os.link(partial, dest)
    partial.unlink()
    return dict(bytes=n, sha256=h.hexdigest())


class Concat:
    """Read-only file-like object over the release parts in order (official: cat part_* > file.tar.gz)."""

    def __init__(self, paths):
        self.paths, self.i, self.f = list(paths), 0, None

    def read(self, n=-1):
        out = bytearray()
        while n < 0 or len(out) < n:
            if self.f is None:
                if self.i >= len(self.paths):
                    break
                self.f = open(self.paths[self.i], 'rb')
                self.i += 1
            block = self.f.read(CHUNK if n < 0 else n - len(out))
            if not block:
                self.f.close()
                self.f = None
                continue
            out += block
        return bytes(out)

    def close(self):
        if self.f is not None:
            self.f.close()


def extract_checkpoint(dest):
    """The one regular member of the concatenated tar.gz, streamed with a running sha256 (no other member accepted)."""
    w = vc.PINS['weights']
    parts = [dest / a['name'] for a in w['assets'] if a['role'] == 'vavim_l_tar_part']
    member_name = w['extracted']['member']
    out = dest / member_name
    partial = out.with_name(out.name + '.partial')
    if out.exists() or partial.exists():
        raise FileExistsError('extracted checkpoint exists; no overwrite')
    src = Concat(parts)
    seen = []
    h, n = hashlib.sha256(), 0
    with tarfile.open(fileobj=src, mode='r|gz') as tar:
        for member in tar:
            seen.append(dict(name=member.name, type=member.type.decode() if isinstance(member.type, bytes) else str(member.type),
                             size=member.size))
            if (member.name != member_name or not member.isreg() or '/' in member.name or member.name.startswith('.')
                    or len(seen) > 1):
                raise ValueError('unexpected tar member: ' + repr(seen))
            reader = tar.extractfile(member)
            with partial.open('xb') as f:
                while True:
                    block = reader.read(CHUNK)
                    if not block:
                        break
                    f.write(block)
                    h.update(block)
                    n += len(block)
    src.close()
    if len(seen) != 1 or n != w['extracted']['bytes'] or (w['extracted'].get('sha256') and h.hexdigest() != w['extracted']['sha256']):
        raise ValueError(f'extracted member mismatch: {seen} {n} {h.hexdigest()} (partial kept)')
    os.link(partial, out)
    partial.unlink()
    return dict(member=member_name, bytes=n, sha256=h.hexdigest(), tar_members=seen,
                rule='cat part_aa part_ab part_ac | tar xzf - (official scripts/handle_checkpoints.py --mode extract), '
                     'streamed with tarfile r|gz; exactly one regular member, no path')


def finish(dest, approval, started, extracted, verify_only):
    """Verify every pinned file (size + sha256), scan the checkpoint pickle, write the records (never overwritten)."""
    w = vc.PINS['weights']
    checked = vc.verify_weights(dest, full=True)
    globals_used = vc.check_pickle_globals(dest / w['extracted']['member'], 'checkpoint')
    sums = [f'{vc.sha(dest / name)}  {name}' for name in [x['name'] for x in w['assets']] + [w['extracted']['member']]]
    for line in sums:
        digest, name = line.split('  ')
        pinned = [x['sha256'] for x in w['assets'] if x['name'] == name] or [w['extracted']['sha256']]
        if digest != pinned[0]:
            raise ValueError('sha256 differs from the pins: ' + name)
    record = dict(schema='wm-calib-bench.vavim.download-verified.v1', release=w['release'], approval=approval,
                  started_utc=started, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                  assets=w['assets'], extracted=extracted or w['extracted'], verified=checked, pickle_globals=globals_used,
                  downloader_sha256=vc.sha(Path(__file__)), pins_sha256=vc.sha(vc.PINS_PATH), verify_only=verify_only,
                  not_downloaded=w['not_downloaded'])
    if not (dest / 'SHA256SUMS').exists():
        with (dest / 'SHA256SUMS').open('x') as f:
            f.write('\n'.join(sums) + '\n')
    if not (dest / 'PROVENANCE.json').exists():
        vc.new_json(dest / 'PROVENANCE.json', dict(source=w['release'], licence=w['license_name'], approval=approval,
                                                  checkpoint_choice=w['checkpoint_choice'],
                                                  files={s.split('  ')[1]: s.split('  ')[0] for s in sums}))
    name = 'download-verified.json'
    if (dest / name).exists():
        name = 'download-verified.verify-' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json'
    vc.new_json(dest / name, record)
    return checked, name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--approval', type=Path, required=True)
    ap.add_argument('--dest', type=Path, required=True)
    ap.add_argument('--verify-only', action='store_true', help='no network: verify an existing download and write the records')
    a = ap.parse_args()
    approval = check_approval(a.approval)             # before any network request or mkdir
    w = vc.PINS['weights']
    if any(not x.get('sha256') for x in w['assets']):
        raise SystemExit('refusing: asset sha256 not pinned')
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
    extracted = None
    if not a.verify_only:
        a.dest.mkdir(parents=True, exist_ok=False)
        for asset in w['assets']:
            got = fetch(asset['url'], a.dest / asset['name'], asset['bytes'], asset['sha256'])
            print(json.dumps(dict(asset=asset['name'], **got)), flush=True)
        extracted = extract_checkpoint(a.dest)
        print(json.dumps(dict(extracted=extracted['member'], bytes=extracted['bytes'], sha256=extracted['sha256'])), flush=True)
        if not w['extracted'].get('sha256') or not w['pickle_globals_allowed'].get('checkpoint'):
            observed = dict(extracted, pickle_globals=vc.pickle_globals(a.dest / w['extracted']['member']), approval=approval,
                            started_utc=started, downloader_sha256=vc.sha(Path(__file__)), pins_sha256=vc.sha(vc.PINS_PATH),
                            note='first extraction: the member sha256 and the pickle globals are observed here; pin them in '
                                 'vavim_pins.json after review, then run --verify-only')
            vc.new_json(a.dest / 'observed-extraction.json', observed)
            print(json.dumps(dict(PIN_REQUIRED=True, observed=str(a.dest / 'observed-extraction.json'))))
            raise SystemExit(3)
    checked, name = finish(a.dest, approval, started, extracted, a.verify_only)
    print(json.dumps(dict(verified=True, files=checked['files'], record=name, dest=str(a.dest), GPU_started=False)))


if __name__ == '__main__':
    main()
