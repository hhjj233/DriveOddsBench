#!/usr/bin/env python3
"""CPU contract tests: grid and canvas map, guards, seeds, I/O, pickle safety, the sampler and the rolling context
against the released code (tiny random MupGPT2), and the runner's window pipeline with stand-in codecs.

  check_cpu.py [--src OFFICIAL_SOURCE] [--snapshot SEED_SNAPSHOT] [--bench BENCH_V1_DIR]   # no GPU, no weights

Tests that need the official source, torch, OpenCV, the seed snapshot or the bench windows are skipped when absent; run it
with the locked inference environment (torch) and with the CPU research environment (OpenCV) for full coverage. No TEST
window or TEST names file is read: TEST guards are exercised with synthetic paths.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')


def _arg(name):
    if name in sys.argv:
        i = sys.argv.index(name)
        value = Path(sys.argv[i + 1]).resolve()
        del sys.argv[i:i + 2]
        return value
    return None


SRC, SNAPSHOT, BENCH = _arg('--src'), _arg('--snapshot'), _arg('--bench')
import numpy as np  # noqa: E402
import vavim_common as vc  # noqa: E402

HAS = {m: importlib.util.find_spec(m) is not None for m in ('torch', 'cv2', 'einops', 'mup', 'lightning', 'hydra')}
OFFICIAL = SRC is not None and all(HAS[m] for m in ('torch', 'einops', 'mup', 'lightning', 'hydra'))


def official():
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import torch
    import mup
    from vam.video_pretraining import mup_gpt2
    return torch, mup, mup_gpt2


def tiny_gpt(seed=0, hw=(18, 32), vocab=16385, nb_timesteps=8):
    torch, mup, mup_gpt2 = official()
    torch.manual_seed(seed)
    gpt = mup_gpt2.MupGPT2(embedding_dim=64, nb_layers=2, dim_heads=16, vocabulary_size=vocab, nb_timesteps=nb_timesteps,
                           nb_tokens_per_timestep=hw[0] * hw[1], init_std=0.5)
    mup.set_base_shapes(gpt, None)
    return gpt.eval().requires_grad_(False)


def stub_engine(gpt):
    """Engine with the tiny GPT and deterministic stand-in codecs (16x16 block mean -> token; token -> flat block)."""
    import torch

    class Enc:
        def __call__(self, x):
            g = torch.nn.functional.avg_pool2d(x.mean(1, keepdim=True), 16)[:, 0]
            return ((g + 1) / 2 * (vc.CODEBOOK - 1)).round().clamp(0, vc.CODEBOOK - 1).to(torch.int64)

    class Dec:
        def __call__(self, t):
            img = (t.float() / (vc.CODEBOOK - 1)) * 2 - 1
            img = img.repeat_interleave(16, 1).repeat_interleave(16, 2)
            return img[:, None].repeat(1, 3, 1, 1)
    return types.SimpleNamespace(torch=torch, gpt=gpt, encoder=Enc(), decoder=Dec(), device=torch.device('cpu'), record={})


class Grid(unittest.TestCase):
    def test_ticks_and_settings(self):
        self.assertEqual(vc.MODEL_WH, vc.CANVAS_WH)
        self.assertEqual(vc.CONTEXT_K, [-20, -15, -10, -5, 0])
        self.assertEqual(vc.FUTURE_K, list(range(5, 61, 5)))
        self.assertEqual(vc.NATIVE2_K, list(range(-20, 61, 5)))
        self.assertEqual(len(vc.NATIVE2_K), 17)
        s = vc.settings()
        self.assertEqual((s['model_fps'], s['canvas_fps'], s['nb_timesteps']), (2, 10, 8))
        self.assertLessEqual(len(vc.CONTEXT_K) + 1, s['nb_timesteps'])

    def test_canvas_map_nearest(self):
        for k in vc.ALL_K:
            m = vc.model_tick(k, 'nearest')
            self.assertEqual(m % 5, 0)
            self.assertLessEqual(abs(m - k), 2)
            others = [x for x in range(-20, 61, 5) if x != m]
            self.assertTrue(all(abs(x - k) > abs(m - k) for x in others))       # strictly nearest: no ties exist
        self.assertEqual([vc.model_tick(k, 'nearest') for k in (-20, -18, -17, -2, -1, 0, 1, 2, 3, 4, 5, 58, 60)],
                         [-20, -20, -15, 0, 0, 0, 0, 0, 5, 5, 5, 60, 60])

    def test_canvas_map_hold(self):
        for k in vc.ALL_K:
            m = vc.model_tick(k, 'hold')
            self.assertEqual(m % 5, 0)
            self.assertTrue(0 <= k - m <= 4)
        self.assertEqual([vc.model_tick(k, 'hold') for k in (-16, -1, 0, 4, 5, 59, 60)], [-20, -5, 0, 0, 5, 55, 60])

    def test_timeline_and_assembly(self):
        rng = np.random.default_rng(0)
        native2 = rng.integers(0, 256, (17, 288, 512, 3), dtype=np.uint8)
        prefix10 = rng.integers(0, 256, (21, 288, 512, 3), dtype=np.uint8)
        prefix10[::5] = native2[:5]
        for rule in ('nearest', 'hold'):
            for rate in ('2hz', '10hz'):
                rows = vc.timeline(rule, rate)
                self.assertEqual([r['canvas_index'] for r in rows], list(range(81)))
                out = vc.assemble_canvas(native2, prefix10, rule, rate)
                self.assertEqual(out.shape, (81, 288, 512, 3))
                for r in rows:
                    want = prefix10[r['k'] + 20] if r['source'] == 'prefix10' else native2[r['native2_index']]
                    np.testing.assert_array_equal(out[r['canvas_index']], want)
                for j, k in enumerate(vc.NATIVE2_K):                       # exact model frames at the 2 Hz ticks
                    np.testing.assert_array_equal(out[k + 20], native2[j])
        rows = vc.timeline('nearest', '2hz')
        self.assertEqual(sum(r['source'] == 'generated' for r in rows), 58)     # ticks 3..60
        self.assertEqual(min(r['k'] for r in rows if r['source'] == 'generated'), 3)
        bad = prefix10.copy()
        bad[20] ^= 1
        with self.assertRaises(ValueError):
            vc.assemble_canvas(native2, bad, 'nearest', '10hz')

    def test_token_guard(self):
        vc.check_tokens(np.array([[0, 16383]]))
        for bad in ([[16384]], [[-1]]):
            with self.assertRaises(ValueError):
                vc.check_tokens(np.array(bad))

    @unittest.skipUnless(HAS['cv2'] and BENCH is not None, 'OpenCV + bench windows')
    def test_render_equals_canvas_real(self):
        import cv2
        w = json.loads((BENCH / 'windows-dev' / '2eeaccb0-t2.5' / 'window.json').read_text())
        row = [r for r in vc.tick_rows(w) if r['k'] == 0][0]
        frame = vc.render(cv2.cvtColor(cv2.imread(row['path']), cv2.COLOR_BGR2RGB))
        real = vc.decode(BENCH / 'windows-dev' / '2eeaccb0-t2.5' / 'canvas' / 'real.mp4', vc.CANVAS_WH, frames=21)
        np.testing.assert_array_equal(frame, real[20])

    @unittest.skipUnless(SNAPSHOT is not None, 'seed snapshot')
    def test_shared_seed_rule(self):
        for w, k in (('2eeaccb0-t2.5', 0), ('47a9fb77-t9.5', 15)):
            self.assertEqual(vc.shared_seed(w, k, SNAPSHOT), vc.local_seed(w, k))
        self.assertNotEqual(vc.local_seed('2eeaccb0-t2.5', 0), vc.local_seed('2eeaccb0-t2.5', 1))


class Guards(unittest.TestCase):
    def test_test_future_guard_before_open(self):
        import build_vavim_inputs as bi
        with patch.object(bi.vc, 'guard_names', side_effect=AssertionError('opened before guard')):
            with self.assertRaises(ValueError):
                bi.build_window('/missing', 'test', 'synthetic', '/missing-out', True, '/missing/protocol-vavim-frozen.txt')
            with self.assertRaises(ValueError):
                bi.build_window('/missing', 'test', 'synthetic', '/missing-out', True, None)

    def test_history_never_selects_future(self):
        t0 = 3_000_000_000
        ticks = [dict(k=k, frame_timestamp_ns=t0 + k * 100_000_000, path=f'f{k}') for k in range(-20, 61)]
        w = dict(ticks=ticks, anchor=dict(timestamp_ns=t0 + 5))
        self.assertEqual([r['k'] for r in vc.tick_rows(w)], list(range(-20, 1)))
        late = copy.deepcopy(w)
        late['ticks'][19]['frame_timestamp_ns'] = t0 + 1
        with self.assertRaises(ValueError):
            vc.tick_rows(late)
        early = copy.deepcopy(w)
        early['anchor']['timestamp_ns'] = t0 - 1
        with self.assertRaises(ValueError):
            vc.tick_rows(early)

    def test_runner_refuses_test_without_record(self):
        import run_vavim_bench as rb
        base = dict(schema='wm-calib-bench.vavim.spec.v1', model='vavim', split='test')
        with self.assertRaises(ValueError):
            rb.validate_spec(dict(base, mode='continue'), '/missing', '/missing')
        with self.assertRaises(ValueError):                      # a generation record does not open TEST references
            rb.validate_spec(dict(base, mode='recon'), '/missing', '/missing', '/tmp/protocol-vavim-frozen.txt')
        with self.assertRaises(ValueError):                      # the right name but no such record
            rb.validate_spec(dict(base, mode='continue'), '/missing', '/missing', '/tmp/x/protocol-vavim-frozen.txt')

    def test_spec_builder_refuses_test_reference(self):
        import make_vavim_specs as ms
        with patch.object(ms.vc, 'guard_names', return_value=None), self.assertRaises(ValueError):
            ms.build('/b', '/i', 'test', ['00000000-t1.0'], 'recon', 'vavim-test-x', [0], '/s', '/o')

    def test_purposes(self):
        with self.assertRaises(ValueError):
            vc.guard_names('/missing', 'test', ['00000000-t1.0'], diagnostic=True)
        ok = [('smoke', 'dev', 'vavim-dev-smoke-v1', 'continue', '2eeaccb0-t2.5', 0),
              ('smoke', 'dev', 'vavim-dev-smoke-v1', 'recon', '47a9fb77-t9.5', None),
              ('diagnostic', 'dev', 'vavim-dev-diag-x', 'continue', '00000000-t1.0', 7),
              ('main', 'dev', 'vavim-dev-k16', 'continue', '00000000-t1.0', 3)]
        for args in ok:
            vc.check_purpose(*args)
        bad = [('smoke', 'test', 'vavim-test-smoke-v1', 'continue', '2eeaccb0-t2.5', 0),
               ('smoke', 'dev', 'vavim-dev-smoke-v1', 'continue', '2eeaccb0-t2.5', 2),
               ('smoke', 'dev', 'vavim-dev-smoke-v1', 'continue', '00000000-t1.0', 0),
               ('diagnostic', 'test', 'vavim-test-diag-x', 'continue', '00000000-t1.0', 0),
               ('main', 'dev', 'vavim-dev-smoke-v1', 'continue', '2eeaccb0-t2.5', 0),
               ('main', 'dev', 'vavim-dev-diag-v1', 'continue', '2eeaccb0-t2.5', 0),
               ('main', 'dev', 'other-dev-k16', 'continue', '2eeaccb0-t2.5', 0)]
        for args in bad:
            with self.assertRaises(ValueError, msg=str(args)):
                vc.check_purpose(*args)

    def test_main_runs_need_rulings(self):
        import run_vavim_bench as rb
        groups = {('dev', 'vavim-dev-k16', 'w'): [('p', dict(purpose='main'))]}
        smoke = {('dev', 'vavim-dev-smoke-v1', 'w'): [('p', dict(purpose='smoke'))]}
        pins = dict(vc.PINS, decisions_pending=['O4'])
        with patch.object(rb.vc, 'PINS', pins):
            with self.assertRaises(ValueError):
                rb.require_rulings(groups)
            rb.require_rulings(smoke)
        with patch.object(rb.vc, 'PINS', dict(vc.PINS, decisions_pending=[])):
            rb.require_rulings(groups)

    def test_download_needs_approval_before_network(self):
        import download_vavim_weights as dw
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / 'weights'
            bad = Path(td) / 'approval.json'
            bad.write_text(json.dumps(dict(model='VaViM', scope='download VaViM-B only', approved_by='owner', licence='x')))
            for approval in (Path(td) / 'missing.json', bad):
                with patch.object(sys, 'argv', ['dl', '--approval', str(approval), '--dest', str(dest)]), \
                        patch.object(dw.urllib.request, 'urlopen', side_effect=AssertionError('network used')):
                    with self.assertRaises(SystemExit):
                        dw.main()
                self.assertFalse(dest.exists())

    def test_download_scope_is_the_pinned_assets(self):
        names = [a['name'] for a in vc.PINS['weights']['assets']]
        self.assertEqual(len(names), 5)
        self.assertTrue(all(n.startswith('width_2048_pretrained_139k_total_155k_chunked') or n.startswith('VQ_ds16_16384_llamagen')
                            for n in names))
        self.assertFalse(any(n.startswith('VAM_') or 'width_768' in n or 'width_1024' in n for n in names))
        self.assertTrue(all(a['url'].startswith('https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/')
                            for a in vc.PINS['weights']['assets']))

    def test_lossless_no_clobber_and_paths(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / 'clip.mp4'
            x = np.random.default_rng(3).integers(0, 256, (5, 24, 32, 3), dtype=np.uint8)
            rec = vc.write_checked(list(x), f, 10)
            self.assertEqual(rec['frames'], 5)
            with self.assertRaises(FileExistsError):
                vc.write_video(x * 0, f, 10)
            self.assertEqual(vc.sha(f), rec['sha256'])
            with self.assertRaises(FileExistsError):
                vc.new_json(Path(td) / 'a.json', {}) or vc.new_json(Path(td) / 'a.json', {})
            with self.assertRaises(ValueError):
                vc.safe_relative(td, '../outside')
            (Path(td) / 'link').symlink_to('/etc')
            with self.assertRaises(ValueError):
                vc.safe_relative(td, 'link/passwd')

    @unittest.skipUnless(HAS['torch'], 'torch')
    def test_pickle_scan(self):
        import pickle
        import torch
        from collections import OrderedDict
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'ok.pt'
            torch.save(OrderedDict(w=torch.zeros(2)), p)
            used = vc.pickle_globals(p)
            self.assertIn(('collections', 'OrderedDict'), used)
            self.assertIn(('torch._utils', '_rebuild_tensor_v2'), used)

            class Evil:
                def __reduce__(self):
                    return (os.system, ('true',))
            q = Path(td) / 'evil.pt'
            torch.save(dict(x=Evil()), q, pickle_protocol=pickle.DEFAULT_PROTOCOL)
            self.assertIn(('posix', 'system'), vc.pickle_globals(q))
            for role in ('checkpoint', 'tokenizer_encoder', 'tokenizer_decoder'):
                with self.assertRaises(ValueError):
                    vc.check_pickle_globals(q, role)

    @unittest.skipUnless(SRC is not None, 'official source')
    def test_source_pins(self):
        rec = vc.verify_source(SRC)
        self.assertEqual(rec['checked_files'], len(vc.PINS['code']['files']))
        text = (SRC / 'vam/video_pretraining/mup_gpt2.py').read_text()
        self.assertIn('context = context[:, -((self.nb_timesteps - 1) * self.nb_tokens_per_timestep) :]', text)   # #57
        self.assertIn('topk_sampler: int = 1,', text)                     # released default: greedy


@unittest.skipUnless(OFFICIAL, 'official source + torch + mup + lightning + hydra')
class Sampler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch, cls.mup, cls.mup_gpt2 = official()
        cls.gpt = tiny_gpt()

    def test_per_row_sampler_equals_released_single_row(self):
        import run_vavim_bench as rb
        torch = self.torch
        logits = torch.randn(1, 3, 16385, generator=torch.Generator().manual_seed(1)) * 3
        for k in (1, 5, 50, 16385):
            for seed in (0, 12345, 2**31 - 1):
                torch.manual_seed(seed)
                released = self.mup_gpt2.MupGPT2._sample_next_token(self.gpt, logits, 1.0, k)
                kit = rb.per_row_sampler(torch, [torch.Generator().manual_seed(seed)])(logits, 1.0, k)
                self.assertTrue(torch.equal(released, kit), (k, seed))

    def test_rows_draw_from_their_own_generator(self):
        import run_vavim_bench as rb
        torch = self.torch
        logits = torch.randn(3, 1, 16385, generator=torch.Generator().manual_seed(2)) * 2
        seeds = [11, 22, 33]
        gens = [torch.Generator().manual_seed(s) for s in seeds]
        batch = [rb.per_row_sampler(torch, gens)(logits, 1.0, 50) for _ in range(20)]
        for row, s in enumerate(seeds):
            g = [torch.Generator().manual_seed(s)]
            alone = [rb.per_row_sampler(torch, g)(logits[row:row + 1], 1.0, 50) for _ in range(20)]
            self.assertTrue(all(torch.equal(b[row:row + 1], x) for b, x in zip(batch, alone)))

    def _oracle(self, burnin, frames, keep_last=True):
        """Independent greedy loop without KV cache: at most 7 context frames, kept from the end (#57) or the start (v1.0.0)."""
        torch = self.torch
        from vam.video_pretraining.prepare_token_sequence import compute_position_indices
        g = self.gpt
        hw = burnin.shape[2] * burnin.shape[3]
        ctx = burnin.reshape(burnin.shape[0], -1)
        pos = compute_position_indices(burnin.shape[0], g.nb_timesteps, burnin.shape[2], burnin.shape[3])
        out = []
        for _ in range(frames):
            if ctx.shape[1] // hw > g.nb_timesteps - 1:
                ctx = ctx[:, -(g.nb_timesteps - 1) * hw:] if keep_last else ctx[:, :(g.nb_timesteps - 1) * hw]
            for _ in range(hw):
                n = ctx.shape[1]
                logits = g.forward(ctx, pos['spatial_positions'][:, :n], pos['temporal_positions'][:, :n], inference=True)
                ctx = torch.cat([ctx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
            out.append(ctx[:, -hw:].reshape(burnin.shape[0], burnin.shape[2], burnin.shape[3]))
        return torch.stack(out, 1)

    def test_rolling_context_keeps_the_last_seven_frames(self):
        torch = self.torch
        hw = (3, 4)
        gpt_small = tiny_gpt(seed=3, hw=hw, vocab=64)
        burnin = torch.randint(0, 64, (1, 5, *hw), generator=torch.Generator().manual_seed(4))
        keep = self.gpt
        self.gpt = gpt_small
        try:
            want = self._oracle(burnin, 6, keep_last=True)
            stale = self._oracle(burnin, 6, keep_last=False)
            for cache in (False, True):
                got = gpt_small.forward_inference(6, burnin, temperature=1.0, topk_sampler=1, use_kv_cache=cache)
                self.assertTrue(torch.equal(got, want), cache)
            self.assertFalse(torch.equal(stale[:, 3:], want[:, 3:]))      # v1.0.0 slicing would condition on stale frames
            self.assertTrue(torch.equal(stale[:, :3], want[:, :3]))       # identical before the first roll (frame 4)
        finally:
            self.gpt = keep

    def test_forward_inference_equivalence_and_batch(self):
        import run_vavim_bench as rb
        torch = self.torch
        engine = stub_engine(self.gpt)
        burnin = torch.randint(0, 16384, (5, 18, 32), generator=torch.Generator().manual_seed(5))
        s = dict(vc.B['sampling'], top_k=vc.B['sampling']['top_k'] or 50)
        with patch.dict(vc.B, {'sampling': s}), tempfile.TemporaryDirectory() as td, torch.inference_mode():
            rec = rb.equivalence_check(engine, burnin, '00000000-t1.0', Path(td) / 'eq.json', 4)   # crosses the first roll
            self.assertTrue(rec['identical_tokens'])
            seeds = [vc.local_seed('00000000-t1.0', k) for k in range(3)]
            with patch.object(vc, 'FUTURE_K', vc.FUTURE_K[:4]):
                batch = rb.generate(engine, burnin, seeds)
                alone = [rb.generate(engine, burnin, [x]) for x in seeds]
            self.assertEqual(tuple(batch.shape), (3, 4, 18, 32))
            self.assertFalse(torch.equal(batch[0], batch[1]))
            # Batch rows use their own generators, but batch numerics (GEMM tiling) differ from batch 1, so a row may
            # leave its batch-1 trajectory after the first token whose top-k set or draw changes; report the agreement.
            first = []
            for i in range(3):
                diff = (batch[i] != alone[i][0]).flatten().nonzero()
                first.append(int(diff[0]) if len(diff) else None)
            print(json.dumps(dict(batch_vs_batch1_first_differing_token=first, tokens_per_row=int(batch[0].numel()))))
            self.assertTrue(all(f is None or f > 0 for f in first))
            self.assertNotIn('_sample_next_token', vars(self.gpt))         # override removed after the call


@unittest.skipUnless(OFFICIAL, 'official source + torch + mup + lightning + hydra')
class Pipeline(unittest.TestCase):
    """run_window end to end with the tiny GPT and stand-in codecs on synthetic inputs (no real window, no weights)."""

    def test_window_outputs(self):
        import run_vavim_bench as rb
        torch = official()[0]
        engine = stub_engine(tiny_gpt(seed=7))
        rng = np.random.default_rng(8)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            root = td / 'inputs' / 'windows' / 'dev' / '00000000-t1.0'
            frames = rng.integers(0, 256, (81, 288, 512, 3), dtype=np.uint8)
            vc.write_checked(list(frames[:21]), root / 'history10.mp4', 10)
            vc.write_checked([frames[k + 20] for k in vc.CONTEXT_K], root / 'context2.mp4', 2)
            vc.write_checked(list(frames), root / 'real10.mp4', 10)
            inputs = {n: dict(path=f'windows/dev/00000000-t1.0/{n}', sha256=vc.sha(root / n))
                      for n in ('history10.mp4', 'context2.mp4', 'real10.mp4')}
            items = []
            for k in (0, 1):
                spec = dict(window='00000000-t1.0', split='dev', run='vavim-dev-smoke-t', mode='continue', sample=k, purpose='smoke',
                            seed=vc.local_seed('00000000-t1.0', k), name=f'00000000-t1.0-s{k}', inputs=inputs,
                            output_dir=f'vavim-dev-smoke-t/native/00000000-t1.0-s{k}')
                items.append((td / f's{k}.json', spec))
            items.append((td / 'r.json', dict(items[0][1], mode='recon', sample=None, seed=None, name='00000000-t1.0-recon',
                                              output_dir='vavim-dev-smoke-t/native/00000000-t1.0-recon')))
            for p, s in items:
                p.write_text(json.dumps(s))
            a = types.SimpleNamespace(input_root=td / 'inputs', output_root=td / 'out')
            s = dict(vc.B['sampling'], top_k=vc.B['sampling']['top_k'] or 50)
            with patch.dict(vc.B, {'sampling': s}), patch.object(vc, 'FUTURE_K', vc.FUTURE_K[:2]), \
                    patch.object(vc, 'NATIVE2_K', vc.NATIVE2_K[:7]), patch.object(vc, 'assemble_canvas', lambda n, p: n), \
                    patch.object(vc, 'settings', lambda: {}), patch.object(vc, 'timeline', lambda *x: []):
                rb.run_window(engine, items, a, dict(common='x'), equivalence=True, equivalence_frames=2)
                out = td / 'out' / 'vavim-dev-smoke-t' / 'native'
                n0 = vc.decode(out / '00000000-t1.0-s0' / 'native2.mp4', vc.MODEL_WH)
                n1 = vc.decode(out / '00000000-t1.0-s1' / 'native2.mp4', vc.MODEL_WH)
                nr = vc.decode(out / '00000000-t1.0-recon' / 'native2.mp4', vc.MODEL_WH)
                self.assertEqual(n0.shape[0], 7)
                np.testing.assert_array_equal(n0[:5], n1[:5])                 # shared prefix
                np.testing.assert_array_equal(n0[:5], nr[:5])
                self.assertFalse(np.array_equal(n0[5:], n1[5:]))              # different seeds, different futures
                rec = json.loads((out / '00000000-t1.0-s1' / 'sample_outputs.json').read_text())
                self.assertEqual((rec['batch']['size'], rec['batch']['row'], rec['seed']), (2, 1, items[1][1]['seed']))
                tok = np.load(out / '00000000-t1.0-recon' / 'tokens.npz')
                self.assertEqual(tok['reference_tokens'].shape, (2, 18, 32))
                self.assertTrue((td / 'out' / 'vavim-dev-smoke-t' / 'equivalence-check-00000000-t1.0.json').is_file())
                with self.assertRaises(FileExistsError):                      # no overwrite, no automatic retry
                    rb.run_window(engine, items, a, dict(common='x'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
