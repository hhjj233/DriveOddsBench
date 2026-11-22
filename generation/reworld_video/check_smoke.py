#!/usr/bin/env python3
"""Check DEV smoke outputs and keep scalar diagnostics; never opens TEST samples.

--set smoke : the 2 registered smoke windows x seeds 0/1 + 1 reference each (6 samples, purpose smoke);
--set tsdiag: the D9 moving subset x seeds 0/1 (8 continuations, purpose timescale-diagnostic; never main DEV).
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
import numpy as np
import reworld_common as rc

COUNTS = {'prefix-native8.mp4': 17, 'pass1-native8.mp4': 33, 'pass2-condition8.mp4': 9, 'pass2-native8.mp4': 33,
          'ref-seg1-input8.mp4': 33, 'ref-seg2-input8.mp4': 33, 'native8.mp4': 65, 'vision.mp4': 81}


def check(sample, expect_settings=True):
    record = json.loads((sample / 'sample_outputs.json').read_text())
    if record.get('split') != 'dev':
        raise ValueError('DEV smoke only; no TEST decoding')
    if record.get('status') != 'success' or record.get('mode') not in ('continue', 'recon'):
        raise ValueError('completed continuation/reference required')
    if expect_settings and record['settings'] != rc.settings():
        raise ValueError('settings drift')
    wh = tuple(record['settings']['size_wh'])
    if record['pass_seeds'] != [record['seed'], record['seed'] + rc.B['pass2_seed_offset']]:
        raise ValueError('seed rule drift')
    if record['mode'] == 'continue' and record['condition_encode_seeds'] != rc.condition_seeds(record['window'], record['sample']):
        raise ValueError('condition seed rule drift')
    rules = record.get('seed_rules') or {}
    if [c['seed'] for c in rules.get('calls', [])][:1] != [record['prefix_seed']] or (
            record['mode'] == 'continue' and [c['seed'] for c in rules['calls'][1:]] != [record['condition_encode_seeds'][0], record['pass_seeds'][0],
                                                                                         record['condition_encode_seeds'][1], record['pass_seeds'][1]]):
        raise ValueError('seed rules/values not recorded (ruling D4)')
    clips = {}
    for name, row in record['files'].items():
        path = rc.safe_relative(sample, name)
        if rc.sha(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('output hash mismatch')
        meta = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-of', 'json', str(path)]))['streams'][0]
        size = (512, 288) if name == 'vision.mp4' else wh
        fps = 10 if name == 'vision.mp4' else 8
        if (meta['width'], meta['height']) != size or meta['r_frame_rate'] != f'{fps}/1':
            raise ValueError('size/fps mismatch: ' + name)
        clips[name] = rc.decode(path, size)
        if len(clips[name]) != COUNTS[name]:
            raise ValueError('frame count mismatch: ' + name)
    prefix, p1, p2, native, vision = [clips[x] for x in ['prefix-native8.mp4', 'pass1-native8.mp4', 'pass2-native8.mp4', 'native8.mp4', 'vision.mp4']]
    if not np.array_equal(native, np.concatenate([prefix[:8], p1, p2[9:]])):
        raise ValueError('native assembly mismatch')
    if not np.array_equal(vision, rc.canvas(native)):
        raise ValueError('8 -> 10 Hz / canvas mapping mismatch')
    if int(vision[:, rc.CANVAS_ROWS[1]:].max()) != 0:
        raise ValueError('uncovered canvas rows are not black')
    if record['mode'] == 'continue' and not np.array_equal(clips['pass2-condition8.mp4'], p1[-9:]):
        raise ValueError('pass 2 was not conditioned on pass 1 frames 24..32')
    strict = record.get('strict_state_dict_loads', [])
    if len(strict) != 2 or any(x['missing_keys'] or x['unexpected_keys'] or not x['official_load_bit_identical'] for x in strict):
        raise ValueError('strict checkpoint loading evidence absent')
    if record['internal_guidance'] != dict(ig_head=None, ig_block_idx=8, ig_scale=1.4, ig_min_timestep=None):
        raise ValueError('self-guided sampling settings drift')

    def jump(i):
        return float(np.abs(vision[i, :rc.CANVAS_ROWS[1]].astype(float) - vision[i - 1, :rc.CANVAS_ROWS[1]]).mean())
    future = native[17:]
    steps = [float(np.abs(future[i].astype(float) - future[i - 1]).mean()) for i in range(1, len(future))]
    return dict(window=record['window'], mode=record['mode'], sample=record['sample'], purpose=record.get('purpose'),
                canvas_sha256=record['files']['vision.mp4']['sha256'],
                format_hashes_assembly_mapping='PASS', t0_to_first_future_canvas_mean_RGB=jump(21), pass_seam_canvas_mean_RGB=jump(51),
                native_future_unique_frames=len({f.tobytes() for f in future}), native_future_step_mean_RGB=float(np.mean(steps)),
                seconds=record['seconds'], pass_wall_times=record['pass_wall_times'], gpu_memory=record.get('gpu_memory'), gpu=record.get('gpu'),
                attention_backend=record.get('attention_backend'), attention_processors=record['attention_processors'], software=record['software'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples', type=Path, nargs='+', required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--set', dest='sample_set', choices=['smoke', 'tsdiag'], default='smoke')
    a = ap.parse_args()
    results = [check(p) for p in a.samples]
    rule = rc.B['smoke'] if a.sample_set == 'smoke' else rc.B['timescale_diagnostic']
    purpose = 'smoke' if a.sample_set == 'smoke' else 'timescale-diagnostic'
    windows = {x['window'] for x in results}
    expected = len(rule['windows']) * (len(rule['samples']) + (1 if rule['references'] else 0))
    if windows != set(rule['windows']) or len(results) != expected or any(x['purpose'] != purpose for x in results):
        raise ValueError(f'{a.sample_set} needs exactly {expected} samples of its registered windows with purpose {purpose}')
    for w in windows:
        rows = [x for x in results if x['window'] == w]
        if {x['sample'] for x in rows if x['mode'] == 'continue'} != set(rule['samples']) or sum(x['mode'] == 'recon' for x in rows) != (1 if rule['references'] else 0):
            raise ValueError('sample set differs')
        if len({x['canvas_sha256'] for x in rows if x['mode'] == 'continue'}) != len(rule['samples']):
            raise ValueError('seed diversity failed: continuation clips byte-identical')
    rc.new_json(a.report, dict(structural_checks='PASS', sample_set=a.sample_set, results=results, visual_continuation_and_A1_pending=True,
                               diagnostics_are_not_quality_gates=True,
                               note=rule.get('rule', 'smoke structural check')))
    print(json.dumps(dict(structural_checks='PASS', sample_set=a.sample_set, samples=len(results), visual_continuation_and_A1_pending=True)))


if __name__ == '__main__':
    main()
