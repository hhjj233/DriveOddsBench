#!/usr/bin/env python3
"""Structural checks of returned DEV smoke samples; scalar diagnostics only (never opens a TEST sample).

  check_smoke.py --samples DIR... --report NEW.json [--equivalence FILE...]

Expects 2 DEV windows x (2 continuations + 1 reference). Checks per sample: hashes, sizes, frame rates and counts,
vision.mp4 == the registered 2 Hz -> 10 Hz map of native2.mp4 (frame for frame), token grids and ranges (no token
>= 16384), settings, strict-load and tokenizer evidence; per window: the shared prefix (native2 frames 0-4 = ticks
-20..0, and every canvas frame shown before the first generated one) is byte-identical in all three clips, the context
token hash agrees, and the two continuations differ. Equivalence records (if given) must report identical tokens.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import vavim_common as vc  # noqa: E402


def probe(path):
    meta = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-of', 'json',
                                               str(path)]))['streams'][0]
    return (meta['width'], meta['height']), meta['r_frame_rate']


def check(sample):
    record = json.loads((sample / 'sample_outputs.json').read_text())
    if record.get('split') != 'dev':
        raise ValueError('DEV smoke only; no TEST decoding')
    if record.get('status') != 'success' or record.get('mode') not in ('continue', 'recon'):
        raise ValueError('completed continuation/reference required')
    if record['settings'] != vc.settings() or record['timeline'] != vc.timeline():
        raise ValueError('settings or canvas map drift')
    for name, row in record['files'].items():
        path = vc.safe_relative(sample, name)
        if vc.sha(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('output hash mismatch: ' + name)
    if probe(sample / 'vision.mp4') != (vc.CANVAS_WH, '10/1') or probe(sample / 'native2.mp4') != (vc.MODEL_WH, '2/1'):
        raise ValueError('size/fps mismatch')
    native2, vision = vc.decode(sample / 'native2.mp4', vc.MODEL_WH), vc.decode(sample / 'vision.mp4', vc.CANVAS_WH)
    if len(native2) != len(vc.NATIVE2_K) or len(vision) != len(vc.ALL_K):
        raise ValueError('frame count mismatch')
    if vc.B['prefix_rate'] == '2hz' and not np.array_equal(vision, vc.assemble_canvas(native2)):
        raise ValueError('canvas mapping mismatch')
    if vc.B['prefix_rate'] == '10hz':
        if not np.array_equal(vision[21:], vc.assemble_canvas(native2, vision[:21])[21:]):
            raise ValueError('canvas mapping mismatch')
    prefix2_sha = hashlib.sha256(np.ascontiguousarray(native2[:5]).tobytes()).hexdigest()
    if prefix2_sha != record['prefix_frames2_sha256']:
        raise ValueError('prefix hash mismatch')
    tok = np.load(sample / 'tokens.npz')
    key = 'generated_tokens' if record['mode'] == 'continue' else 'reference_tokens'
    if (tok['history_tokens'].shape != (21, *vc.TOKEN_HW) or tok['context_tokens'].shape != (len(vc.CONTEXT_K), *vc.TOKEN_HW)
            or tok[key].shape != (len(vc.FUTURE_K), *vc.TOKEN_HW)):
        raise ValueError('token grid shapes')
    if any(int(tok[k].min()) < 0 or int(tok[k].max()) >= vc.CODEBOOK for k in ('history_tokens', 'context_tokens', key)):
        raise ValueError('token range')
    if not np.array_equal(tok['context_tokens'], tok['history_tokens'][[k + 20 for k in vc.CONTEXT_K]]):
        raise ValueError('context tokens are not the history tokens at the context ticks')
    ctx_sha = hashlib.sha256(tok['context_tokens'].astype(np.int16).tobytes()).hexdigest()
    if ctx_sha != record['context_tokens_sha256']:
        raise ValueError('context token hash mismatch')
    load = record['load']
    if load['strict_state_dict_load']['missing_keys'] or load['strict_state_dict_load']['unexpected_keys'] or \
            load['mup_base_shapes']['lm_head_width_mult'] != vc.B['lm_head_width_mult'] or not load['mup_base_shapes']['agrees_with_saved_infshapes']:
        raise ValueError('strict loading / readout evidence absent')
    first_gen = min(r['canvas_index'] for r in record['timeline'] if r['source'] == 'generated')
    shown_prefix = hashlib.sha256(np.ascontiguousarray(vision[:first_gen]).tobytes()).hexdigest()
    steps = [float(np.abs(native2[j].astype(float) - native2[j - 1]).mean()) for j in range(1, len(native2))]
    out = dict(window=record['window'], mode=record['mode'], sample=record['sample'], seed=record['seed'],
               canvas_sha256=record['files']['vision.mp4']['sha256'], prefix2_sha256=prefix2_sha, shown_prefix_sha256=shown_prefix,
               context_tokens_sha256=ctx_sha, structural='PASS',
               native2_step_mean_abs_RGB=dict(history=steps[:4], t0_to_first=steps[4], future=steps[5:]),
               seconds=record['seconds'], gpu_memory=record['gpu_memory'], gpu=record['gpu'], batch=record.get('batch'),
               generated_token_stats=record.get('generated_token_stats'), software=record.get('software'))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--samples', type=Path, nargs='+', required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--equivalence', type=Path, nargs='*', default=[])
    a = ap.parse_args()
    results = [check(p) for p in a.samples]
    windows = sorted({x['window'] for x in results})
    if len(windows) != 2 or len(results) != 6:
        raise ValueError('smoke needs exactly 2 DEV windows x 2 continuations plus 2 references')
    for w in windows:
        rows = [x for x in results if x['window'] == w]
        if sorted(x['sample'] for x in rows if x['mode'] == 'continue') != [0, 1] or sum(x['mode'] == 'recon' for x in rows) != 1:
            raise ValueError('smoke sample set differs')
        for key in ('prefix2_sha256', 'shown_prefix_sha256', 'context_tokens_sha256'):
            if len({x[key] for x in rows}) != 1:
                raise ValueError(f'shared prefix differs between the clips of {w} ({key})')
        if len({x['canvas_sha256'] for x in rows if x['mode'] == 'continue'}) != 2:
            raise ValueError('seed diversity failed: both continuation clips byte-identical')
    eq = [json.loads(p.read_text()) for p in a.equivalence]
    if any(not e['identical_tokens'] for e in eq):
        raise ValueError('released/kit equivalence failed')
    vc.new_json(a.report, dict(structural_checks='PASS', results=results, equivalence=eq,
                               visual_review_and_A1_pending=True, diagnostics_are_not_quality_gates=True))
    print(json.dumps(dict(structural_checks='PASS', samples=len(results), equivalence_checks=len(eq))))


if __name__ == '__main__':
    main()
