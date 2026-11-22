#!/usr/bin/env python3
"""Synthetic guard/rollout/geometry tests. No model imports, GPUs or real TEST windows."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode = True
import numpy as np
import reworld_common as rc
import build_reworld_inputs as bi
import download_reworld_weights as dw
import make_reworld_specs as ms
import run_reworld_bench as runner


def synthetic_window(t0=3_000_000_000):
    return dict(frames_20hz=[dict(timestamp_ns=t, path='synthetic') for t in range(0, 10_000_000_000, 50_000_000)],
                ticks=[dict(k=0, frame_timestamp_ns=t0)], anchor=dict(timestamp_ns=t0))


class Contracts(unittest.TestCase):
    def test_grid(self):
        rows = rc.timeline()
        self.assertEqual(len(rows), 81)
        self.assertEqual(max(abs(r['native_minus_canvas_ms']) for r in rows), 50)
        for k in (0, 10, 20, 40, 60):
            self.assertEqual(rc.native_index(k), 16 + 4 * k // 5)
        future = [rc.native_index(k) for k in range(61)]
        self.assertEqual(sum(a == b for a, b in zip(future, future[1:])), 12)
        self.assertEqual(rc.nearest([0, 50], 25), 0)

    def test_history_never_selects_future(self):
        w = synthetic_window()
        history = rc.camera_rows(w)
        self.assertEqual(len(history), 17)
        self.assertTrue(all(r['timestamp_ns'] <= 3_000_000_000 for r in history))
        full = rc.camera_rows(w, with_future=True)
        self.assertEqual([r['timestamp_ns'] for r in full[:17]], [r['timestamp_ns'] for r in history])
        gap = copy.deepcopy(w)
        gap['frames_20hz'] = [r for r in gap['frames_20hz'] if not 3_000_000_000 - 160_000_000 < r['timestamp_ns'] < 3_000_000_000 - 90_000_000]
        with self.assertRaises(ValueError):
            rc.camera_rows(gap)

    def test_settings_and_geometry(self):
        s = rc.settings()
        self.assertEqual(s['size_wh'][0], 2 * s['size_wh'][1])
        self.assertEqual(s['latent_grid'], [5, s['size_wh'][1] // 32, s['size_wh'][0] // 32])
        self.assertEqual(rc.SRC_ROWS[1] - rc.SRC_ROWS[0], rc.SRC_W // 2)
        self.assertEqual(rc.SRC_ROWS[0], rc.B['band_rows'][0])
        # canvas scale s = 512/1550 maps the source crop exactly onto the valid canvas rows
        s_canvas = rc.CANVAS_W / rc.SRC_W
        self.assertAlmostEqual((rc.SRC_ROWS[0] - rc.B['band_rows'][0]) * s_canvas, rc.CANVAS_ROWS[0], places=9)
        self.assertAlmostEqual((rc.SRC_ROWS[1] - rc.B['band_rows'][0]) * s_canvas, rc.CANVAS_ROWS[1], places=9)
        self.assertEqual(rc.CANVAS_ROWS[1] - rc.CANVAS_ROWS[0], rc.CANVAS_W // 2)

    def test_render_and_canvas_alignment(self):
        import cv2
        rng = np.random.default_rng(3)
        raw = rng.integers(0, 256, (rc.SRC_H, rc.SRC_W, 3), dtype=np.uint8)
        raw[1200:1210] = 255  # a bright horizontal bar at source rows [1200, 1210)
        model = rc.render_model_frame(raw)
        self.assertEqual(model.shape, (rc.H, rc.W, 3))
        canvas = rc.to_canvas(model)
        self.assertEqual(canvas.shape, (288, 512, 3))
        self.assertEqual(int(canvas[rc.CANVAS_ROWS[1]:].max()), 0)
        direct = cv2.resize(raw[rc.SRC_ROWS[0]:rc.SRC_ROWS[1]], (512, rc.CANVAS_ROWS[1] - rc.CANVAS_ROWS[0]), interpolation=cv2.INTER_AREA)
        diff = np.abs(canvas[rc.CANVAS_ROWS[0]:rc.CANVAS_ROWS[1]].astype(int) - direct.astype(int))
        self.assertLess(diff.mean(), 1.0)  # two-stage vs direct area resampling of the same source rows
        rows = np.where(canvas[:, :, 0].mean(axis=1) > 200)[0]
        expected = ((1200 - 797) * 512 / 1550, (1210 - 797) * 512 / 1550)
        self.assertTrue(expected[0] - 1 <= rows.min() and rows.max() <= expected[1] + 1, (rows, expected))

    def test_canvas_tick_selection(self):
        native = np.zeros((65, rc.H, rc.W, 3), np.uint8)
        for j in range(65):
            native[j] = j
        vision = rc.canvas(native)
        self.assertEqual(vision.shape, (81, 288, 512, 3))
        for i, k in enumerate(range(-20, 61)):
            self.assertTrue((vision[i, :rc.CANVAS_ROWS[1]] == rc.native_index(k)).all())
            self.assertEqual(int(vision[i, rc.CANVAS_ROWS[1]:].max()), 0)

    def test_seeds(self):
        w = '2eeaccb0-t2.5'
        self.assertEqual(rc.aux_seed(w, 3), int.from_bytes(hashlib.sha256(f'bench-v1|{w}|3'.encode()).digest()[:4], 'big') & 0x7fffffff)
        c = rc.condition_seeds(w, 3)
        self.assertEqual(len(set(c + [rc.aux_seed(w, 3), rc.aux_seed(w, 3) + 1000003, rc.aux_seed(w, 'prefix')])), 5)
        self.assertNotEqual(rc.condition_seeds(w, 3), rc.condition_seeds(w, 4))

    def test_pass2_uses_generated_context(self):
        calls, saved, written = [], [], []
        history = np.full((9, 2, 4, 3), 240, np.uint8)

        def fake(engine, path, expected, seed, cond_seed):
            calls.append((path, expected.copy(), seed, cond_seed))
            return np.broadcast_to(np.arange(33, dtype=np.uint8)[:, None, None, None], (33, 2, 4, 3)).copy()

        def write2(frames):
            written.append(frames.copy())
            return 'pass2-condition8.mp4'
        with patch.object(runner, 'official_pass', fake):
            p1, p2 = runner.continuation_passes(None, 'cond1.mp4', history, [7, 1000010], [11, 13], lambda x: saved.append(x.copy()), write2)
        self.assertEqual(len(saved), 1)
        np.testing.assert_array_equal(written[0], p1[-9:])
        np.testing.assert_array_equal(calls[1][1], p1[-9:])
        self.assertEqual([c[0] for c in calls], ['cond1.mp4', 'pass2-condition8.mp4'])
        self.assertEqual([c[2] for c in calls], [7, 1000010])
        self.assertEqual([c[3] for c in calls], [11, 13])
        with self.assertRaises(ValueError):
            runner.continuation_passes(None, 'cond1.mp4', history, [7, 8], [11, 13], saved.append, write2)

    def test_licence_before_network_and_mkdir(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / 'weights'
            with patch.object(sys, 'argv', ['download', '--dest', str(dest)]), \
                    patch.object(dw.urllib.request, 'urlopen', side_effect=AssertionError('network used')):
                with self.assertRaises(ValueError):
                    dw.main()
            self.assertFalse(dest.exists())

    def test_licence_scope(self):
        repos = []
        for key, repo in rc.PINS['weights']['repos'].items():
            lic = next(x for x in rc.PINS['weights']['files'] if x['repo_key'] == key and x['repo_path'] == repo['license_path'])
            repos.append(dict(repo=repo['repo'], revision=repo['revision'], license_sha256=lic['sha256']))
        with tempfile.TemporaryDirectory() as td:
            ok = Path(td) / 'ok.json'
            ok.write_text(json.dumps(dict(owner_approved=True, approved_by='owner', repos=repos)))
            dw.license_ok(ok)
            bad = Path(td) / 'bad.json'
            wrong = copy.deepcopy(repos)
            wrong[1]['revision'] = '0' * 40
            bad.write_text(json.dumps(dict(owner_approved=True, approved_by='owner', repos=wrong)))
            with self.assertRaises(ValueError):
                dw.license_ok(bad)

    def test_TEST_future_guard_before_open(self):
        with patch.object(bi.rc, 'guard_names', side_effect=AssertionError('split/window opened before guard')):
            with self.assertRaises(ValueError):
                bi.build_window('/missing', 'test', 'synthetic', '/missing-out', True, '/missing-freeze')

    def test_TEST_reference_spec_needs_freeze(self):
        with tempfile.TemporaryDirectory() as td:
            bench = Path(td)
            (bench / 'names-test-gen.txt').write_text('0123abcd-t1.5\n')
            with self.assertRaises(ValueError):
                ms.build(bench, bench, 'test', ['0123abcd-t1.5'], 'recon', 'reworld-test-ref-v1', ['recon'], bench, bench / 'specs')
            self.assertFalse((bench / 'specs').exists())

    def test_runner_requires_coordinator_rulings(self):
        with patch.dict(rc.B, {'coordinator_approved': False}):
            with self.assertRaises(ValueError):
                rc.require_approved()
        with patch.dict(rc.B, {'coordinator_approved': True}), patch.dict(rc.PINS, {'decisions_pending': [dict(id='D0')]}):
            with self.assertRaises(ValueError):
                rc.require_approved()
        with patch.dict(rc.B, {'coordinator_approved': True}), patch.dict(rc.PINS, {'decisions_pending': []}):
            rc.require_approved()

    def test_rulings_recorded(self):
        self.assertEqual(rc.PINS['decisions_pending'], [])
        self.assertEqual([d['id'] for d in rc.PINS['decisions']], ['D' + str(i) for i in range(1, 12)])
        self.assertTrue(all(d['by'] == 'coordinator p-inter-84' and d['date'] == '2026-11-13' and d['choice'] for d in rc.PINS['decisions']))
        self.assertIs(rc.B['coordinator_approved'], True)
        self.assertEqual((rc.B['coordinator_approved_by'], rc.B['coordinator_approved_date']), ('coordinator p-inter-84', '2026-11-13'))
        rc.require_approved()
        self.assertEqual(rc.B['size_wh'], [1024, 512])
        self.assertEqual(rc.B['geometry']['model_source_rows'], [797, 1572])

    def test_purposes(self):
        sm, dg = rc.B['smoke'], rc.B['timescale_diagnostic']
        rc.check_purpose('smoke', 'dev', 'reworld-dev-smoke-v2', 'continue', sm['windows'][0], 1)
        rc.check_purpose('smoke', 'dev', 'reworld-dev-smoke-ref-v2', 'recon', sm['windows'][1], 'recon')
        rc.check_purpose('timescale-diagnostic', 'dev', 'reworld-dev-tsdiag-v1', 'continue', dg['windows'][2], 0)
        rc.check_purpose('main', 'dev', 'reworld-dev-v1', 'continue', dg['windows'][1], 7)
        bad = [('timescale-diagnostic', 'dev', 'reworld-dev-tsdiag-v1', 'recon', dg['windows'][0], 'recon'),
               ('timescale-diagnostic', 'dev', 'reworld-dev-tsdiag-v1', 'continue', dg['windows'][0], 2),
               ('timescale-diagnostic', 'dev', 'reworld-dev-v1', 'continue', dg['windows'][0], 0),
               ('timescale-diagnostic', 'dev', 'reworld-dev-tsdiag-v1', 'continue', sm['windows'][0], 0),
               ('smoke', 'dev', 'reworld-dev-smoke-v2', 'continue', dg['windows'][0], 0),
               ('smoke', 'test', 'reworld-test-smoke-v2', 'continue', sm['windows'][0], 0),
               ('main', 'dev', 'reworld-dev-tsdiag-v1', 'continue', dg['windows'][0], 0),
               ('main', 'dev', 'reworld-dev-smoke-v9', 'continue', sm['windows'][0], 0),
               (None, 'dev', 'reworld-dev-v1', 'continue', sm['windows'][0], 0)]
        for args in bad:
            with self.assertRaises(ValueError, msg=str(args)):
                rc.check_purpose(*args)

    def test_window_lists_by_purpose(self):
        reg = dict(file='names-dev-gen.txt', sha256=rc.B['registered_names_sha256']['dev'])
        view = dict(file='names-dev-gen.txt', sha256=rc.B['timescale_diagnostic']['bench_view_names_dev_gen_sha256'])
        rc.check_names_list('main', 'dev', reg)
        rc.check_names_list('smoke', 'dev', reg)
        rc.check_names_list('timescale-diagnostic', 'dev', view)
        for purpose, listed in (('main', view), ('smoke', view), ('timescale-diagnostic', reg), ('main', None)):
            with self.assertRaises(ValueError):
                rc.check_names_list(purpose, 'dev', listed)

    def test_runner_requires_absolute_paths(self):
        with patch.object(sys, 'argv', ['run', '--specs', 'relative.json', '--bench', '/b', '--input-root', '/i', '--output-root', '/o',
                                        '--src', '/s', '--weights', '/w', '--dry-run']), \
                patch.object(runner.rc, 'verify_source', side_effect=AssertionError('read before the path check')):
            with self.assertRaises(ValueError):
                runner.main()

    def test_validate_spec_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'names-dev-gen.txt').write_text('0123abcd-t1.5\n')
            inputs = {}
            for kind in ('history', 'condition'):
                p = root / 'in' / (bi.clip_name(kind))
                p.parent.mkdir(exist_ok=True)
                p.write_bytes(kind.encode())
                inputs[bi.clip_name(kind)] = dict(path='in/' + p.name, sha256=rc.sha(p))
            w, k = '0123abcd-t1.5', 2
            seed = rc.aux_seed(w, k)
            spec = dict(schema='wm-calib-bench.reworld_video.spec.v1', model='reworld_video', split='dev', window=w, mode='continue',
                        run='reworld-dev-x', name=w + '-s2', sample=k, seed=seed, pass_seeds=[seed, seed + 1000003],
                        condition_encode_seeds=rc.condition_seeds(w, k), prefix_seed=rc.aux_seed(w, 'prefix'), inputs=inputs,
                        prompt=None, purpose='main', names_list=rc.names_list(root, 'dev'), settings=rc.settings(),
                        output_dir='reworld-dev-x/native/' + w + '-s2', generator=ms.code_provenance())
            with self.assertRaises(ValueError):  # a synthetic window list is neither registered nor the diagnostic view
                runner.validate_spec(copy.deepcopy(spec), root, root)
            with patch.dict(rc.B, {'registered_names_sha256': {'dev': spec['names_list']['sha256'], 'test': '0' * 64}}):
                self.check_tampering(spec, seed, root)

    def check_tampering(self, spec, seed, root):
        runner.validate_spec(copy.deepcopy(spec), root, root)
        for key, value in (('pass_seeds', [seed, seed + 1]), ('seed', seed + 1), ('prompt', 'drive'),
                           ('condition_encode_seeds', [1, 2]), ('prefix_seed', 5), ('purpose', 'smoke'),
                           ('purpose', 'timescale-diagnostic'), ('purpose', None), ('run', 'reworld-dev-tsdiag-x'),
                           ('names_list', dict(file='names-dev-gen.txt', sha256='0' * 64))):
            bad = copy.deepcopy(spec)
            bad[key] = value
            with self.assertRaises(ValueError, msg=key):
                runner.validate_spec(bad, root, root)
        bad = copy.deepcopy(spec)
        bad['settings']['num_inference_steps'] = 50
        with self.assertRaises(ValueError):
            runner.validate_spec(bad, root, root)
        (root / 'in' / bi.clip_name('history')).write_bytes(b'changed')
        with self.assertRaises(ValueError):
            runner.validate_spec(copy.deepcopy(spec), root, root)

    def test_lossless_no_clobber(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / 'clip.mp4'
            x = np.random.default_rng(1).integers(0, 256, (5, 24, 32, 3), dtype=np.uint8)
            rc.write_video(x, f, 8)
            np.testing.assert_array_equal(rc.decode(f, (32, 24)), x)
            digest = rc.sha(f)
            with self.assertRaises(FileExistsError):
                rc.write_video(x * 0, f, 8)
            self.assertEqual(rc.sha(f), digest)

    def test_path_escape(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                rc.safe_relative(td, '../outside')
            with self.assertRaises(ValueError):
                rc.safe_relative(td, '/absolute')
            (Path(td) / 'link').symlink_to('/etc')
            with self.assertRaises(ValueError):
                rc.safe_relative(td, 'link/passwd')


if __name__ == '__main__':
    unittest.main(verbosity=2)
