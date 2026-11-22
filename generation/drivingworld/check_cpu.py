#!/usr/bin/env python3
"""CPU contract tests: geometry, poses, guards, I/O and the rollout against the released code (tiny random GPT).

  check_cpu.py [--src OFFICIAL_SOURCE]      # no GPU, no weights, no real TEST window

Tests that need the official source, torch, OpenCV or SciPy are skipped when absent; run it with both the CPU research
environment (OpenCV/SciPy) and the locked inference environment (torch 2.1.0) for full coverage.
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
SRC = None
if '--src' in sys.argv:
    i = sys.argv.index('--src')
    SRC = Path(sys.argv[i + 1]).resolve()
    del sys.argv[i:i + 2]
import numpy as np  # noqa: E402
import drivingworld_common as dc  # noqa: E402

HAS = {m: importlib.util.find_spec(m) is not None for m in ('torch', 'cv2', 'scipy', 'einops')}
OFFICIAL = SRC is not None and HAS['torch'] and HAS['einops']


def official_import():
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import torch
    torch.Tensor.cuda = lambda self, *a, **k: self      # the released modules call .cuda() at construction
    torch.nn.Module.cuda = lambda self, *a, **k: self
    from models.model import TrainTransformers
    return torch, TrainTransformers


def tiny_model(torch, TrainTransformers, condition_frames=3):
    args = types.SimpleNamespace(codebook_size=64, codebook_embed_dim=8, image_size=[32, 64], downsample_size=16, pkeep=0.7,
                                 pose_x_vocab_size=128, pose_y_vocab_size=128, yaw_vocab_size=512, n_layer=[1, 1], n_embd=64,
                                 gpt_type='ar')
    torch.manual_seed(0)
    tw = TrainTransformers(args, local_rank=0, condition_frames=condition_frames).eval()
    return tw, torch.nn.Embedding(64, 8)


class FakeTokenizer:
    """Stand-in for the VQ decoder: the frame shows the newest token grid (bookkeeping tests only)."""
    def __init__(self, torch, codebook):
        self.torch = torch
        self.vq_model = types.SimpleNamespace(quantize=types.SimpleNamespace(embedding=codebook))

    def z_to_clips(self, indices):
        last = indices[-1].float() / 64
        img = last.repeat_interleave(16, 0).repeat_interleave(16, 1)[None, None].repeat(1, 3, 1, 1)
        return img.clamp(0, 1)


class Grid(unittest.TestCase):
    def test_timeline(self):
        rows = dc.timeline()
        self.assertEqual(len(rows), 81)
        self.assertEqual([r['canvas_index'] for r in rows], list(range(81)))
        self.assertEqual(rows[20]['k'], 0)
        self.assertEqual(sum(r['source'] == 'generated' for r in rows), 60)
        self.assertEqual(dc.CONDITION_K, list(range(-14, 1)))

    def test_bands(self):
        c, t = dc.band('centred'), dc.band('top')
        self.assertEqual((c['canvas_offset'], c['canvas_rows'], c['valid_canvas_rows']), (72, [-53, 203], [0, 203]))
        self.assertEqual((t['canvas_offset'], t['canvas_rows'], t['valid_canvas_rows']), (0, [0, 256], [0, 256]))
        frames = np.random.default_rng(0).integers(0, 256, (2, 256, 512, 3), dtype=np.uint8)
        out = dc.to_canvas(frames, c)
        np.testing.assert_array_equal(out[:, :203], frames[:, 53:256])
        self.assertEqual(int(out[:, 203:].max()), 0)
        out = dc.to_canvas(frames, t)
        np.testing.assert_array_equal(out[:, :256], frames)
        self.assertEqual(int(out[:, 256:].max()), 0)
        bad = copy.deepcopy(dc.B)
        bad['bands']['centred']['tall_rows'] = [18, 275]
        with patch.object(dc, 'B', bad), self.assertRaises(ValueError):
            dc.band('centred')

    @unittest.skipUnless(HAS['cv2'], 'OpenCV')
    def test_render_shares_canvas_grid(self):
        rgb = np.random.default_rng(1).integers(0, 256, (2048, 1550, 3), dtype=np.uint8)
        for name in ('centred', 'top'):
            g = dc.band(name)
            model, canvas, tall = dc.render(rgb, g)
            np.testing.assert_array_equal(tall[g['canvas_offset']:g['canvas_offset'] + 288], canvas)
            v0, v1 = g['valid_canvas_rows']
            np.testing.assert_array_equal(dc.to_canvas(model[None], g)[0, v0:v1], canvas[v0:v1])

    def test_pose_tokens_and_integration(self):
        ix, iy, iyaw = dc.pose_tokens_np([[0.0, 0.0], [1.5, 0.1], [-0.2, -0.7], [9.0, 0.6]], [0.0, 1.0, -9.0, 8.5])
        self.assertEqual(ix.tolist(), [0, 24, 0, 127])
        self.assertEqual(iy.tolist(), [63, 76, 0, 127])
        self.assertEqual(iyaw.tolist(), [255, 287, 0, 511])
        if HAS['torch']:
            from drivingworld_rollout import integrate
            np.testing.assert_allclose(integrate([[0, 1, 0]] * 3)[-1], [3, 0, 0])
            np.testing.assert_allclose(integrate([[90, 1, 0], [0, 1, 0]])[-1], [1, 1, 90], atol=1e-12)

    @unittest.skipUnless(HAS['scipy'], 'SciPy')
    def test_rel_poses(self):
        from scipy.spatial.transform import Rotation
        rng = np.random.default_rng(2)
        mats = np.tile(np.eye(4), (15, 1, 1))
        mats[:, :3, :3] = Rotation.from_euler('zyx', rng.normal(0, [0.3, 0.02, 0.02], (15, 3))).as_matrix()
        mats[:, :3, 3] = rng.normal(0, 5, (15, 3))
        xy, yaw = dc.rel_poses(mats)
        np.testing.assert_allclose(np.r_[xy[0], yaw[0]], 0, atol=1e-9)    # inv(P0) P0: identity up to round-off, as released
        rel = np.linalg.inv(mats[3]) @ mats[4]
        np.testing.assert_allclose(xy[4], rel[:2, 3])
        if SRC is not None and HAS['cv2']:
            sys.path.insert(0, str(SRC))
            from datasets.datasets_utils import get_meta_data
            ref = get_meta_data(poses=list(mats))
            np.testing.assert_array_equal(xy, ref['rel_poses'])
            np.testing.assert_array_equal(yaw, ref['rel_yaws'])


class Guards(unittest.TestCase):
    def test_test_future_guard_before_open(self):
        import build_drivingworld_inputs as bi
        with patch.object(bi.dc, 'guard_names', side_effect=AssertionError('opened before guard')):
            with self.assertRaises(ValueError):
                bi.build_window('/missing', 'test', 'synthetic', '/missing-out', 'centred', True, '/missing-freeze')

    def test_history_never_selects_future(self):
        t0 = 3_000_000_000
        ticks = [dict(k=k, frame_timestamp_ns=t0 + k * 100_000_000, path=f'f{k}') for k in range(-20, 61)]
        w = dict(ticks=ticks, anchor=dict(timestamp_ns=t0 + 5))
        rows = dc.tick_rows(w)
        self.assertEqual([r['k'] for r in rows], list(range(-20, 1)))
        late = copy.deepcopy(w)
        late['ticks'][19]['frame_timestamp_ns'] = t0 + 1
        with self.assertRaises(ValueError):
            dc.tick_rows(late)
        early = copy.deepcopy(w)
        early['anchor']['timestamp_ns'] = t0 - 1
        with self.assertRaises(ValueError):
            dc.tick_rows(early)

    def test_runner_refuses_test_without_record(self):
        import run_drivingworld_bench as rb
        with self.assertRaises(ValueError):
            rb.validate_spec(dict(schema='wm-calib-bench.drivingworld.spec.v1', model='drivingworld', split='test', mode='continue'),
                             '/missing', '/missing')
        with self.assertRaises(ValueError):
            rb.validate_spec(dict(schema='wm-calib-bench.drivingworld.spec.v1', model='drivingworld', split='test', mode='recon'),
                             '/missing', '/missing', '/tmp/protocol-drivingworld-frozen.txt')

    def test_spec_builder_refuses_test_reference(self):
        import make_drivingworld_specs as ms
        with patch.object(ms.dc, 'guard_names', return_value=None), self.assertRaises(ValueError):
            ms.build('/b', '/i', 'test', ['00000000-t1.0'], 'recon', 'drivingworld-test-x', [0], '/s', '/o')

    def test_diagnostic_windows_are_dev_only(self):
        import make_drivingworld_specs as ms
        import run_drivingworld_bench as rb
        with self.assertRaises(ValueError):
            dc.guard_names('/missing', 'test', ['00000000-t1.0'], diagnostic=True)
        with patch.object(ms.dc, 'guard_names', return_value=None), self.assertRaises(ValueError):
            ms.build('/b', '/i', 'dev', ['00000000-t1.0'], 'continue', 'drivingworld-dev-smoke', [0], '/s', '/o', diagnostic=True)
        with self.assertRaises(ValueError):
            rb.validate_spec(dict(schema='wm-calib-bench.drivingworld.spec.v1', model='drivingworld', split='dev', mode='continue',
                                  diagnostic=True, run='drivingworld-dev-smoke-v1'), '/missing', '/missing')

    def test_download_needs_approval_before_network(self):
        import download_drivingworld_weights as dw
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / 'weights'
            with patch.object(sys, 'argv', ['dl', '--dest', str(dest)]), \
                    patch.object(dw.urllib.request, 'urlopen', side_effect=AssertionError('network used')):
                with self.assertRaises(ValueError):
                    dw.main()
            self.assertFalse(dest.exists())
            bad = Path(td) / 'approval.json'
            bad.write_text(json.dumps(dict(owner_approved=True, approved_by='owner', repo=dc.PINS['weights']['repo'],
                                           revision='main', license='MIT')))
            with patch.object(sys, 'argv', ['dl', '--dest', str(dest), '--approval', str(bad)]), \
                    patch.object(dw.urllib.request, 'urlopen', side_effect=AssertionError('network used')):
                with self.assertRaises(ValueError):
                    dw.main()
            self.assertFalse(dest.exists())

    def test_lossless_no_clobber_and_paths(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / 'clip.mp4'
            x = np.random.default_rng(3).integers(0, 256, (5, 24, 32, 3), dtype=np.uint8)
            rec = dc.write_checked(list(x), f, 10)
            self.assertEqual(rec['frames'], 5)
            with self.assertRaises(FileExistsError):
                dc.write_video(x * 0, f, 10)
            self.assertEqual(dc.sha(f), rec['sha256'])
            with self.assertRaises(ValueError):
                dc.safe_relative(td, '../outside')
            (Path(td) / 'link').symlink_to('/etc')
            with self.assertRaises(ValueError):
                dc.safe_relative(td, 'link/passwd')

    @unittest.skipUnless(HAS['torch'], 'torch')
    def test_pickle_scan(self):
        import pickle
        import torch
        from collections import OrderedDict
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'ok.pth'
            torch.save(OrderedDict(w=torch.zeros(2)), p)
            used = dc.pickle_globals(p)
            self.assertIn(('collections', 'OrderedDict'), used)
            self.assertIn(('torch._utils', '_rebuild_tensor_v2'), used)

            class Evil:
                def __reduce__(self):
                    return (os.system, ('true',))
            q = Path(td) / 'evil.pth'
            torch.save(dict(x=Evil()), q, pickle_protocol=pickle.DEFAULT_PROTOCOL)
            self.assertIn(('posix', 'system'), dc.pickle_globals(q))
            with self.assertRaises(ValueError):
                dc.check_pickle_globals(q, 'world_model.pth')


@unittest.skipUnless(OFFICIAL, 'official source + torch + einops')
class Rollout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch, TT = official_import()
        cls.tw, cls.codebook = tiny_model(cls.torch, TT)
        import drivingworld_rollout as dr
        cls.dr = dr

    def inputs(self, B=1, seed=4):
        torch = self.torch
        g = torch.Generator().manual_seed(seed)
        tokens = torch.randint(0, 64, (B, 3, 8), generator=g)
        pose = torch.rand((B, 3, 2), generator=g) * torch.tensor([3.0, 0.4]) - torch.tensor([0.0, 0.2])
        yaw = torch.rand((B, 3, 1), generator=g) * 4 - 2
        pose[:, 0] = 0
        yaw[:, 0] = 0
        return tokens, pose, yaw

    def test_pose_roundtrip_and_numpy_copy(self):
        torch = self.torch
        from modules.tokenizers.pose_tokenizer import poses_to_indices, yaws_to_indices, indices_to_pose, indices_to_yaws
        i = torch.arange(128)[None, :]
        x, y = indices_to_pose(i, i)
        back = poses_to_indices(torch.stack([x, y], -1).float(), 128, 128)
        self.assertTrue(torch.equal(back[..., 0], i) and torch.equal(back[..., 1], i))
        j = torch.arange(512)[None, :, None]
        self.assertTrue(torch.equal(yaws_to_indices(indices_to_yaws(j).float(), 512), j))
        vals = np.random.default_rng(5).uniform([-1, -1, -12], [10, 1, 12], (500, 3)).astype(np.float32)
        ix, iy, iyaw = dc.pose_tokens_np(vals[:, :2], vals[:, 2])
        t = poses_to_indices(torch.from_numpy(vals[None, :, :2]), 128, 128)[0]
        self.assertEqual(t[:, 0].tolist(), ix.tolist())
        self.assertEqual(t[:, 1].tolist(), iy.tolist())
        self.assertEqual(yaws_to_indices(torch.from_numpy(vals[None, :, 2:]), 512)[0, :, 0].tolist(), iyaw.tolist())

    def test_sampler_matches_released_and_rows_are_independent(self):
        torch = self.torch
        from utils.generat_utils import top_k_sampling
        logits = torch.randn(3, 1, 300, generator=torch.Generator().manual_seed(6))
        for b in range(3):
            torch.manual_seed(100 + b)
            ref = top_k_sampling(logits[b:b + 1].clone(), top_k=30, temperature_k=1.0)
            got = self.dr.topk_rows(logits[b:b + 1].clone(), 30, 1.0, [torch.Generator().manual_seed(100 + b)])
            self.assertTrue(torch.equal(ref, got))
        rows = self.dr.topk_rows(logits, 30, 1.0, [torch.Generator().manual_seed(100 + b) for b in range(3)])
        singles = torch.cat([self.dr.topk_rows(logits[b:b + 1], 30, 1.0, [torch.Generator().manual_seed(100 + b)]) for b in range(3)])
        self.assertTrue(torch.equal(rows, singles))
        draws = torch.cat([self.dr.topk_rows(logits[:1], 30, 1.0, [torch.Generator().manual_seed(s)]) for s in range(400)])
        allowed = set(torch.topk(logits[0, 0], 30).indices.tolist())
        self.assertTrue(set(draws[:, 0].tolist()) <= allowed and len(set(draws[:, 0].tolist())) > 10)

    def test_internal_hidden_reproduces_eval_third(self):
        torch = self.torch
        gpt = self.tw.model
        tokens, pose, yaw = self.inputs(B=2)
        from modules.tokenizers.pose_tokenizer import poses_to_indices, yaws_to_indices
        flag = torch.isinf(pose[:, 0, 0])
        with torch.no_grad():
            intra = gpt.eval_first_second(self.codebook(tokens), poses_to_indices(pose, 128, 128), yaws_to_indices(yaw, 512), flag)
            yaws = torch.tensor([[[250]], [[260]]])
            xs = torch.tensor([[[20]], [[3]]])
            poses = torch.cat([xs, torch.tensor([[[60]], [[70]]])], -1)
            new = self.codebook(torch.tensor([[5, 9], [1, 2]]))
            cases = [(intra[:, :1], {}, 'yaw_logits', gpt.yaw_head, slice(0, 1)),
                     (intra[:, :2], dict(yaws=yaws), 'pose_x_logits', gpt.pose_x_head, slice(1, 2)),
                     (intra[:, :3], dict(yaws=yaws, poses=xs), 'pose_y_logits', gpt.pose_x_head, slice(2, 3)),
                     (intra[:, :6], dict(yaws=yaws, poses=poses, new_tokens=new), 'img_logits', gpt.head, slice(3, None))]
            for emb, kw, key, head, sl in cases:
                ref = gpt.eval_third(emb, drop_flag=flag, **kw)[key]
                got = head(self.dr.internal_hidden(gpt, emb, drop_flag=flag, **kw)[:, sl])
                self.assertTrue(torch.equal(ref, got), key)
            # the kit's y logits use the training head, which the released eval_third does not
            y = self.dr.pose_y_logits(gpt, intra[:, :3], yaws, xs, flag)
            h = self.dr.internal_hidden(gpt, intra[:, :3], yaws=yaws, poses=xs, drop_flag=flag)
            self.assertTrue(torch.equal(y, gpt.pose_y_head(h[:, 2:3])))
            self.assertFalse(torch.equal(y, gpt.eval_third(intra[:, :3], yaws=yaws, poses=xs, drop_flag=flag)['pose_y_logits']))

    def test_image_loop_matches_released_generate(self):
        torch = self.torch
        from modules.tokenizers.pose_tokenizer import indices_to_pose, indices_to_yaws
        tokens, pose, yaw = self.inputs(B=1)
        feature = self.codebook(tokens)
        given = (torch.tensor([[262]]), torch.tensor([[17]]), torch.tensor([[66]]))
        gx, gy = indices_to_pose(given[1], given[2])
        gyaw = indices_to_yaws(given[0])
        with torch.no_grad():
            torch.manual_seed(77)
            ref = self.tw.generate_gt_pose_gt_yaw(tokens, feature, pose, yaw, torch.stack([gx, gy], -1).float(), gyaw[:, :, None].float(),
                                                  self.codebook, sampling_mtd='top_k', temperature_k=1.0, top_k=30)
            got = self.dr.sample_next_state(self.tw, feature, pose, yaw, self.codebook, [torch.Generator().manual_seed(77)],
                                            dict(method='top_k', top_k=30, temperature=1.0), dict(method='top_k', top_k=30, temperature=1.0),
                                            given=given)
        self.assertTrue(torch.equal(ref, got[3]))
        self.assertEqual([int(v) for v in got[:3]], [262, 17, 66])

    def test_rollout_bookkeeping(self):
        torch = self.torch
        tokens, pose, yaw = self.inputs(B=1)
        calls = []
        real = self.dr.sample_next_state

        def spy(tw, feature, pose_win, yaw_win, codebook, gens, pcfg, icfg, given=None):
            calls.append((pose_win.clone(), yaw_win.clone(), feature.clone()))
            return real(tw, feature, pose_win, yaw_win, codebook, gens, pcfg, icfg, given=given)
        cfg = dict(method='top_k', top_k=30, temperature=1.0)
        tok = FakeTokenizer(torch, self.codebook)
        with patch.object(self.dr, 'sample_next_state', spy), torch.no_grad():
            out = self.dr.rollout(self.tw, tok, tokens, pose[0].numpy(), yaw[0].numpy(), [11, 12], 4, cfg, cfg, autocast=False)
        self.assertEqual(out['tokens'].shape, (2, 4, 8))
        self.assertEqual(out['frames'].shape, (2, 4, 32, 64, 3))
        for t, (pw, yw, feat) in enumerate(calls):
            self.assertTrue(bool((pw[:, 0] == 0).all()) and bool((yw[:, 0] == 0).all()))
            if t:
                np.testing.assert_allclose(pw[:, -1].numpy(), out['pose_values'][:, t - 1, 1:], rtol=0, atol=1e-6)
                np.testing.assert_allclose(yw[:, -1, 0].numpy(), out['pose_values'][:, t - 1, 0], rtol=0, atol=1e-6)
                self.assertTrue(torch.equal(feat[:, -1], self.codebook(torch.from_numpy(out['tokens'][:, t - 1]))))
        np.testing.assert_allclose(calls[1][0][0, 1:-1].numpy(), pose[0, 2:].numpy())
        # the fake decoder shows the newest grid: frames follow the sampled tokens, so decoding used the shifted window
        grid = (out['frames'][0, 2, ::16, ::16, 0].astype(float) / 255 * 64).round()
        np.testing.assert_allclose(grid, out['tokens'][0, 2].reshape(2, 4), atol=1)
        with patch.object(self.dr, 'sample_next_state', spy), torch.no_grad():
            again = self.dr.rollout(self.tw, tok, tokens, pose[0].numpy(), yaw[0].numpy(), [11], 4, cfg, cfg, autocast=False)
        self.assertTrue(np.array_equal(again['tokens'][0], out['tokens'][0]))


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    unittest.main(verbosity=2)
