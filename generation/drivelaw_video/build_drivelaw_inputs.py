#!/usr/bin/env python3
"""Raw AV2 -> native8Hz DriveLaW inputs, with strict history-only motion fields."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import drivelaw_common as dc

def historical_motion(window):
    """Filter pose rows before materialisation: nothing after the anchor is read."""
    import numpy as np
    import pyarrow.dataset as ds
    from av2.geometry.geometry import quat_to_mat
    from av2.geometry.se3 import SE3
    anchor=int(window['anchor']['timestamp_ns'])
    camera0=next(x for x in window['ticks'] if x['k']==0)
    log=Path(camera0['path']).parents[3]
    posefile=log/'city_SE3_egovehicle.feather'
    table=ds.dataset(str(posefile),format='feather').to_table(
        filter=(ds.field('timestamp_ns')>=anchor-2_100_000_000)&(ds.field('timestamp_ns')<=anchor))
    data=table.to_pandas().sort_values('timestamp_ns')
    ts=[int(x) for x in data.timestamp_ns]
    if not ts or ts[-1]!=anchor:raise ValueError('exact anchor ego pose required; no future interpolation')
    rotations=quat_to_mat(data[['qw','qx','qy','qz']].to_numpy())
    translations=data[['tx_m','ty_m','tz_m']].to_numpy()
    transforms=[SE3(rotation=r,translation=t) for r,t in zip(rotations,translations)]
    targets=[anchor-2_000_000_000,anchor-1_000_000_000,anchor-500_000_000,anchor]
    indices=[dc.nearest(ts,t) for t in targets]
    if any(abs(ts[i]-t)>50_000_000 for i,t in zip(indices,targets)):raise ValueError('history pose gap exceeds50ms')
    inv=transforms[-1].inverse();rel=inv.compose(transforms[indices[0]])
    i1,ih,it=indices[1:]
    def velocity(a,z):
        dt=(ts[z]-ts[a])/1e9
        if dt<=0:raise ValueError('history derivative interval empty')
        return (translations[z,:2]-translations[a,:2])/dt
    current=velocity(ih,it);previous=velocity(i1,ih)
    speed=float(np.linalg.norm(current))
    accel=float(np.linalg.norm((current-previous)/((ts[it]-ts[i1])/2e9)))
    return dict(anchor_ns=anchor,latest_pose_ns=ts[-1],history_interval_s=2.,
                forward_m=float(rel.translation[0]),lateral_m=float(rel.translation[1]),
                yaw_rad=float(np.arctan2(rel.rotation[1,0],rel.rotation[0,0])),
                speed_mps=speed,accel_mps2=accel,
                derivative_rule='speed=norm of planar velocity over last0.5s; acceleration=norm of difference between last and previous0.5s planar velocities / mean interval; actual timestamp durations; past-only finite differences',
                pose_source=str(posefile),pose_source_sha256=dc.sha(posefile),
                selected_poses=[dict(target_ns=t,timestamp_ns=ts[i],offset_ns=ts[i]-t) for i,t in zip(indices,targets)],
                past_pose_rows_loaded=len(ts),future_pose_rows_loaded=0,navigation_command_read=False)

def build_window(bench,split,window_name,out,with_future=False,frozen_file=None):
    if split=='test' and with_future:
        dc.verify_freeze(frozen_file,required=[dc.PINS_PATH,Path(__file__)])
    dc.guard_names(bench,split,[window_name])
    source=Path(bench)/('windows-'+split)/window_name/'window.json'
    window=json.loads(source.read_text())
    if window['name']!=window_name or window['split'].lower()!=split:raise ValueError('window split mismatch')
    # No frame is opened until the split/future guard and all timestamps pass.
    rows=dc.camera_rows(window,with_future)
    import cv2
    import numpy as np
    cv2.setNumThreads(2)
    dest=Path(out)/'windows'/split/window_name
    if dest.exists():raise FileExistsError('fresh window destination required; existing outputs retained')
    dest.mkdir(parents=True,exist_ok=False)
    frames=[]
    for row in rows:
        rgb=cv2.cvtColor(cv2.imread(row['path']),cv2.COLOR_BGR2RGB)
        if list(rgb.shape[:2])!=[2048,1550]:raise ValueError('raw AV2 camera shape changed')
        frames.append(cv2.resize(rgb[797:1669],(1280,704),interpolation=cv2.INTER_AREA))
    clips={'history8_1280.mp4':frames[:17],'condition8_1280.mp4':frames[8:17]}
    if with_future:clips['real8_1280.mp4']=frames
    files={}
    for name,value in clips.items():
        path=dest/name;dc.write_video(value,path,8);decoded=dc.decode(path,(1280,704))
        if not np.array_equal(decoded,np.stack(value)):raise ValueError('native input lossless decode mismatch')
        files[name]=dict(sha256=dc.sha(path),bytes=path.stat().st_size,frames=len(value),fps=8,size_wh=[1280,704])
    motion=historical_motion(window)
    dc.new_json(dest/'history_motion.json',motion)
    files['history_motion.json']=dict(sha256=dc.sha(dest/'history_motion.json'),bytes=(dest/'history_motion.json').stat().st_size)
    record=dict(schema='wm-calib-bench.drivelaw_video.inputs.v1',model='drivelaw_video',split=split,window=window_name,
                source_window_sha256=dc.sha(source),raw_camera_rows=rows,
                history_camera_rows=[r for r in rows if r['native_index']<=16],condition_camera_rows=rows[8:17],
                max_camera_error_ms=max(abs(r['error_ns'])/1e6 for r in rows),
                t0_camera_ns=rows[16]['timestamp_ns'],anchor_ns=int(window['anchor']['timestamp_ns']),
                geometry=dict(band_rows=[797,1669],native_wh=[1280,704],canvas_wh=[512,288],render='cv2.INTER_AREA'),
                mapping=dc.timeline(),files=files,future_present=with_future,
                generator_sha256=dc.sha(Path(__file__)),common_sha256=dc.sha(dc.HERE/'drivelaw_common.py'),pins_sha256=dc.sha(dc.PINS_PATH))
    dc.new_json(dest/'inputs.json',record)
    return dict(window=window_name,directory=str(dest),files=len(files),history_frames=17,condition_frames=9,real_frames=len(frames) if with_future else 0,max_camera_error_ms=record['max_camera_error_ms'])

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--bench',type=Path,required=True);ap.add_argument('--split',choices=['dev','test'],required=True);ap.add_argument('--names',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--allow-test-future',action='store_true');ap.add_argument('--frozen-file',type=Path);a=ap.parse_args()
    if a.split=='test' and a.allow_test_future:dc.verify_freeze(a.frozen_file or '',required=[dc.PINS_PATH,Path(__file__)])
    windows=dc.names(a.names);dc.guard_names(a.bench,a.split,windows)
    result=[build_window(a.bench,a.split,w,a.out,a.split=='dev' or a.allow_test_future,a.frozen_file) for w in windows]
    dc.new_json(a.out/('build-'+a.split+'.json'),dict(complete=True,split=a.split,windows=result,TEST_future_allowed=a.allow_test_future))
    print(json.dumps(dict(complete=True,windows=len(result),out=str(a.out),GPU_started=False)))
if __name__=='__main__':main()
