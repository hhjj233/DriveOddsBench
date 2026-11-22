#!/usr/bin/env python3
"""Synthetic guard/rollout tests. No model imports, GPUs or real TEST windows."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
import numpy as np
import drivelaw_common as dc
import build_drivelaw_inputs as bi
import download_drivelaw_weights as dw
import run_drivelaw_bench as runner

class Contracts(unittest.TestCase):
    def test_grid(self):
        rows=dc.timeline();self.assertEqual(len(rows),81)
        self.assertEqual(max(abs(r['native_minus_canvas_ms']) for r in rows),50)
        for k in (0,10,20,40,60):self.assertEqual(dc.native_index(k),16+4*k//5)
        future=[dc.native_index(k) for k in range(61)]
        self.assertEqual(sum(a==b for a,b in zip(future,future[1:])),12)
        self.assertEqual(dc.nearest([0,50],25),0)

    def test_history_never_selects_future(self):
        t0=3_000_000_000
        w=dict(frames_20hz=[dict(timestamp_ns=t,path='synthetic') for t in range(0,10_000_000_000,50_000_000)],
               ticks=[dict(k=0,frame_timestamp_ns=t0)],anchor=dict(timestamp_ns=t0))
        history=dc.camera_rows(w);self.assertEqual(len(history),17)
        self.assertTrue(all(r['timestamp_ns']<=t0 for r in history))
        gap=copy.deepcopy(w);gap['frames_20hz']=[r for r in gap['frames_20hz'] if not t0-160_000_000<r['timestamp_ns']<t0-90_000_000]
        with self.assertRaises(ValueError):dc.camera_rows(gap)

    def test_prompt(self):
        m=dict(forward_m=-20.,lateral_m=-1.,yaw_rad=-.1,speed_mps=5.,accel_mps2=.5,anchor_ns=100,latest_pose_ns=100)
        s=dc.caption(m);self.assertIn('past 2 seconds',s);self.assertIn('next 4 seconds',s)
        self.assertIn('at moderate speed',s);self.assertIn('gradually changing speed',s)
        self.assertIn('forward 20.00m',s)
        for word in ('turn','route','left','right','straight'):self.assertNotIn(word,s)
        m['latest_pose_ns']=101
        with self.assertRaises(ValueError):dc.caption(m)
        m['latest_pose_ns']=100;m['speed_mps']=float('nan')
        with self.assertRaises(ValueError):dc.caption(m)

    def test_pass2_uses_generated_context(self):
        calls=[];saved=[];history=np.full((9,2,2,3),240,np.uint8)
        def fake(engine,context,prompt,seed):
            calls.append((context.copy(),prompt,seed))
            return np.broadcast_to(np.arange(33,dtype=np.uint8)[:,None,None,None],(33,2,2,3)).copy()
        with patch.object(runner,'official_pass',fake):
            p1,p2=runner.continuation_passes(None,history,'fixed prompt',[7,1000010],lambda x:saved.append(x.copy()))
        self.assertEqual(len(saved),1);np.testing.assert_array_equal(calls[1][0],p1[-9:])
        self.assertFalse(np.array_equal(calls[1][0],history))
        self.assertEqual([x[1] for x in calls],['fixed prompt']*2)
        self.assertEqual([x[2] for x in calls],[7,1000010])

    def test_licence_before_network_and_mkdir(self):
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/'weights'
            with patch.object(sys,'argv',['download','--dest',str(dest)]),patch.object(dw.urllib.request,'urlopen',side_effect=AssertionError('network used')):
                with self.assertRaises(ValueError):dw.main()
            self.assertFalse(dest.exists())

    def test_TEST_future_guard_before_open(self):
        with patch.object(bi.dc,'guard_names',side_effect=AssertionError('split/window opened before guard')):
            with self.assertRaises(ValueError):bi.build_window('/missing','test','synthetic','/missing-out',True,'/missing-freeze')

    def test_lossless_no_clobber(self):
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'clip.mp4';x=np.random.default_rng(1).integers(0,256,(5,24,32,3),dtype=np.uint8)
            dc.write_video(x,f,8);np.testing.assert_array_equal(dc.decode(f,(32,24)),x)
            digest=dc.sha(f)
            with self.assertRaises(FileExistsError):dc.write_video(x*0,f,8)
            self.assertEqual(dc.sha(f),digest)

    def test_path_escape(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):dc.safe_relative(td,'../outside')
            with self.assertRaises(ValueError):dc.safe_relative(td,'/absolute')
            (Path(td)/'link').symlink_to('/etc')
            with self.assertRaises(ValueError):dc.safe_relative(td,'link/passwd')

if __name__=='__main__':unittest.main(verbosity=2)
