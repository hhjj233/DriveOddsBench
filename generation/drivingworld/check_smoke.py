#!/usr/bin/env python3
"""Structural checks of returned DEV smoke samples; scalar diagnostics only (never opens a TEST sample).

  check_smoke.py --samples DIR... --report NEW.json [--equivalence FILE...]

Expects 2 DEV windows x (2 continuations + 1 reference). Checks per sample: hashes, sizes, frame rates and counts,
canvas == registered mapping of native10, token/pose ranges, settings and strict-load evidence; per window: the shared
prefix (canvas frames 0-20) is byte-identical in all three clips and the two continuations differ.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import drivingworld_common as dc  # noqa: E402


def check(sample):
    record = json.loads((sample / 'sample_outputs.json').read_text())
    if record.get('split') != 'dev':
        raise ValueError('DEV smoke only; no TEST decoding')
    if record.get('status') != 'success' or record.get('mode') not in ('continue', 'recon'):
        raise ValueError('completed continuation/reference required')
    if record['settings'] != dc.settings() or record['band'] != dc.band(record['settings']['band']):
        raise ValueError('settings drift')
    clips = {}
    for name, row in record['files'].items():
        path = dc.safe_relative(sample, name)
        if dc.sha(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('output hash mismatch: ' + name)
        if name.endswith('.mp4'):
            meta = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams',
                                                       '-of', 'json', str(path)]))['streams'][0]
            wh = dc.CANVAS_WH if name == 'vision.mp4' else dc.MODEL_WH
            if (meta['width'], meta['height']) != wh or meta['r_frame_rate'] != '10/1':
                raise ValueError('size/fps mismatch: ' + name)
            clips[name] = dc.decode(path, wh)
    native, vision = clips['native10.mp4'], clips['vision.mp4']
    if len(native) != 81 or len(vision) != 81:
        raise ValueError('frame count mismatch')
    if not np.array_equal(vision, dc.to_canvas(native, record['band'])):
        raise ValueError('canvas mapping mismatch')
    prefix_sha = hashlib.sha256(np.ascontiguousarray(native[:21]).tobytes()).hexdigest()
    if prefix_sha != record['prefix_frames_sha256']:
        raise ValueError('prefix hash mismatch')
    tok = np.load(sample / 'tokens.npz')
    if tok['cond_tokens_a'].shape != (16, 512) or tok['cond_tokens_b'].shape != (16, 512):
        raise ValueError('conditioning token shapes')
    key = 'generated_tokens' if record['mode'] == 'continue' else 'reference_tokens'
    if tok[key].shape != (60, 512) or tok[key].min() < 0 or tok[key].max() >= 16384:
        raise ValueError('token range/shape')
    out = dict(window=record['window'], mode=record['mode'], sample=record['sample'], seed=record['seed'],
               canvas_sha256=record['files']['vision.mp4']['sha256'], prefix_sha256=prefix_sha, structural='PASS',
               t0_to_first_future_canvas_mean_abs_RGB=float(np.abs(vision[21].astype(float) - vision[20]).mean()),
               future_step_mean_abs_RGB=float(np.mean([np.abs(vision[i].astype(float) - vision[i - 1]).mean() for i in range(22, 81)])),
               seconds=record['seconds'], gpu_memory=record['gpu_memory'], gpu=record['gpu'], batch=record.get('batch'),
               strict_state_dict_loads=record['strict_state_dict_loads'])
    if len(record.get('strict_state_dict_loads', [])) != 2 or record['strict_state_dict_loads'][0]['missing_keys']:
        raise ValueError('strict loading evidence absent')
    if record['mode'] == 'continue':
        pt = tok['pose_tokens_yaw_x_y']
        if pt.shape != (60, 3) or pt[:, 0].max() >= 512 or pt[:, 1:].max() >= 128 or pt.min() < 0:
            raise ValueError('pose token range/shape')
        poses = json.loads((sample / 'pred_poses.json').read_text())
        if len(poses['rel_yaw_deg_x_m_y_m']) != 60 or not np.isfinite(np.array(poses['integrated_x_y_heading'])).all():
            raise ValueError('predicted poses malformed')
        hist = poses['condition_xy_m'][-1]
        out.update(first_step_pred_x_m=poses['rel_yaw_deg_x_m_y_m'][0][1], history_last_x_m=hist[0],
                   pred_end_xy_m=poses['integrated_x_y_heading'][-1][:2],
                   pred_speed_mps_0_1s=float(np.hypot(*poses['integrated_x_y_heading'][9][:2]) / 1.0))
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
        if len({x['prefix_sha256'] for x in rows}) != 1:
            raise ValueError('shared prefix differs between the clips of ' + w)
        if len({x['canvas_sha256'] for x in rows if x['mode'] == 'continue'}) != 2:
            raise ValueError('seed diversity failed: both continuation clips byte-identical')
    eq = [json.loads(p.read_text()) for p in a.equivalence]
    if any(not e['identical_tokens'] for e in eq):
        raise ValueError('released/kit equivalence failed')
    dc.new_json(a.report, dict(structural_checks='PASS', results=results, equivalence=eq,
                               visual_review_and_A1_pending=True, diagnostics_are_not_quality_gates=True))
    print(json.dumps(dict(structural_checks='PASS', samples=len(results), equivalence_checks=len(eq))))


if __name__ == '__main__':
    main()
