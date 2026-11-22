"""DriveLaW bench helpers: strict provenance, past-only inputs and exact grids."""
from __future__ import annotations
import bisect
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

HERE=Path(__file__).resolve().parent
PINS_PATH=HERE/'drivelaw_pins.json'
PINS=json.loads(PINS_PATH.read_text())
B=PINS['bench']
NS=1_000_000_000

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()

def blob(path):
    p=Path(path);h=hashlib.sha1(b'blob '+str(p.stat().st_size).encode()+b'\0')
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()

def new_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def safe_relative(root,name):
    rel=Path(name)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('relative input path required')
    root=Path(root).resolve();p=root/rel
    if not p.resolve().is_relative_to(root) or any(x.is_symlink() for x in [p,*p.parents] if x.is_relative_to(root)):
        raise ValueError('input path escapes root or uses symlink')
    return p

def names(path):
    values=Path(path).read_text().split()
    if len(values)!=len(set(values)) or any(not re.fullmatch(r'[0-9a-f]{8}-t[0-9]+\.[0-9]+',v) for v in values):
        raise ValueError('unique bench window names required')
    return values

def nearest(timestamps,target):
    """Nearest integer timestamp; ties go to the earlier observation."""
    i=bisect.bisect_left(timestamps,target)
    candidates=[j for j in (i-1,i) if 0<=j<len(timestamps)]
    if not candidates:raise ValueError('empty timestamp source')
    return min(candidates,key=lambda j:(abs(timestamps[j]-target),timestamps[j]))

def native_index(k):
    """10Hz tick k=-20..60 -> nearest native8Hz frame j=0..64, t0 at16."""
    if k not in range(-20,61):raise ValueError('canvas tick outside registered range')
    q=Fraction(16)+Fraction(4*k,5);lo=q.numerator//q.denominator
    return min((lo,lo+1),key=lambda j:(abs(Fraction(j)-q),j))

def timeline():
    return [dict(k=k,native8_index=native_index(k),native_minus_canvas_ms=float((Fraction(native_index(k)-16,8)-Fraction(k,10))*1000)) for k in range(-20,61)]

def camera_rows(window,with_future=False):
    """Read metadata only. History requests never select/read a future image."""
    src=window['frames_20hz'];timestamps=[int(x['timestamp_ns']) for x in src]
    if timestamps!=sorted(set(timestamps)):raise ValueError('camera timestamps must be strictly increasing')
    tick0=next(x for x in window['ticks'] if x['k']==0)
    t0=int(tick0['frame_timestamp_ns']);anchor=int(window['anchor']['timestamp_ns'])
    if t0>anchor:raise ValueError('t0 camera image is later than benchmark anchor')
    offsets=range(-16,49) if with_future else range(-16,1)
    rows=[]
    for offset in offsets:
        target=t0+offset*125_000_000
        eligible=timestamps if with_future else [t for t in timestamps if t<=t0]
        j=nearest(eligible,target);stamp=eligible[j];index=bisect.bisect_left(timestamps,stamp)
        if abs(stamp-target)>25_100_000:raise ValueError('raw20Hz nearest-frame error exceeds25.1ms')
        rows.append(dict(native_index=offset+16,relative_s=offset/8,target_timestamp_ns=target,
                         timestamp_ns=stamp,error_ns=stamp-target,path=src[index]['path']))
    if any(r['timestamp_ns']>t0 for r in rows if r['native_index']<=16):raise ValueError('future image in history')
    if len({r['timestamp_ns'] for r in rows})!=len(rows):raise ValueError('duplicate raw conditioning timestamps')
    return rows

def verify_freeze(record,required=()):
    """SHA records only; call before any TEST future or TEST sample is read."""
    record=Path(record)
    if not record.is_file():raise ValueError('coordinator freeze record required')
    checked={};pending=[record];visited=set()
    while pending:
        current=pending.pop()
        if current in visited:continue
        visited.add(current);found=0
        for line in current.read_text().splitlines():
            match=re.fullmatch(r'([0-9a-f]{64})\s+(.+?)\s*',line.strip())
            if not match:continue
            digest,name=match.groups();target=Path(name)
            if not target.is_absolute():target=current.parent/target
            if sha(target)!=digest:raise ValueError('frozen hash mismatch: '+str(target))
            checked[str(target.resolve())]=digest;found+=1
            if target.name.endswith(('-files.sha256','-frozen.txt')):pending.append(target)
        if not found:raise ValueError('empty or unsupported freeze record')
    if any(str(Path(p).resolve()) not in checked for p in required):raise ValueError('required runner/pins absent from freeze')
    return dict(path=str(record),sha256=sha(record),entries=len(checked))

def guard_names(bench,split,windows):
    if split not in ('dev','test'):raise ValueError('split must be dev or test')
    allowed=set(names(Path(bench)/('names-'+split+'-gen.txt')))
    if not windows or len(windows)!=len(set(windows)) or not set(windows)<=allowed:
        raise ValueError('window set differs from registered split')

def settings():
    if B['temporal_frame_rate'] is None or B['prompt_past_seconds'] is None or B['prompt_future_seconds'] is None:
        raise ValueError('temporal frame rate or caption labels missing from drivelaw_pins.json')
    return dict(size_hw=[704,1280],native_fps=8,canvas_fps=10,condition_frames=9,num_frames=33,
                num_inference_steps=30,guidance_scale=3,temporal_frame_rate=B['temporal_frame_rate'],
                passes=2,pass2_seed_offset=1000003,reuse_pass1_prompt=True,transition_anchor=True,
                anchor_strength=1.,noise_reinjection_enabled=True,noise_reinjection_beta=1.,
                noise_reinjection_sigma=.3,noise_reinjection_steps=3,dtype='bfloat16',VAE_tiling=True)

def caption(motion):
    settings()
    values=[motion[k] for k in ('forward_m','lateral_m','yaw_rad','speed_mps','accel_mps2')]
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values):raise ValueError('finite historical motion required')
    if motion['latest_pose_ns']>motion['anchor_ns']:raise ValueError('future pose in caption')
    speed='at low speed' if motion['speed_mps']<5 else 'at moderate speed' if motion['speed_mps']<15 else 'at highway speed'
    stability='steady motion' if abs(motion['accel_mps2'])<.5 else 'gradually changing speed'
    # Navigation-command and turn/route wording are deliberately absent.
    return ('A high-quality, photorealistic dashboard camera view of autonomous driving. '
            f"Based on the past {B['prompt_past_seconds']:g} seconds, predict and generate the next {B['prompt_future_seconds']:g} seconds of realistic driving continuation, "
            f"moving {speed} with {stability}. "
            'Maintain temporal consistency, stable camera perspective, natural motion flow without jitter or artifacts, '
            'clear details, and realistic physics. '
            f"[Technical: forward {abs(motion['forward_m']):.2f}m, lateral {abs(motion['lateral_m']):.2f}m, "
            f"yaw {abs(math.degrees(motion['yaw_rad'])):.1f}°, speed {motion['speed_mps']:.2f}m/s]")

def shared_seed(window,sample,snapshot):
    root=Path(snapshot);path=root/PINS['seed_snapshot']['file'];expected=PINS['seed_snapshot']['sha256']
    listed={name:digest for digest,name in [line.split('  ',1) for line in (root/'MANIFEST.sha256').read_text().splitlines() if '  ' in line]}
    if sha(path)!=expected or listed.get(PINS['seed_snapshot']['file'])!=expected:raise ValueError('shared seed source hash mismatch')
    spec=importlib.util.spec_from_file_location('drivelaw_seed_snapshot',path);module=importlib.util.module_from_spec(spec)
    old=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    try:spec.loader.exec_module(module)
    finally:sys.dont_write_bytecode=old
    value=module.seed_of(window,sample)
    local=int.from_bytes(hashlib.sha256(f'bench-v1|{window}|{sample}'.encode()).digest()[:4],'big')&0x7fffffff
    if value!=local:raise ValueError('shared seed rule drift')
    return value

def verify_source(root):
    root=Path(root).resolve()
    registered=dict(PINS['code']['files'])
    registered.update(PINS['code'].get('build_files',{}))
    reconstructed=PINS['code'].get('reconstructed_models',{})
    registered.update(reconstructed.get('files',{}))
    for name,digest in registered.items():
        p=safe_relative(root,name)
        if not p.is_file() or blob(p)!=digest:raise ValueError('official inference source changed: '+name)
    expected={str(Path(n).relative_to('DriveLaW-Video/Infer/diffusers/src/diffusers')) for n in registered if '/diffusers/src/diffusers/' in n and n.endswith('.py')}
    path=root/'DriveLaW-Video/Infer/diffusers/src/diffusers'
    actual={str(x.relative_to(path)) for x in path.rglob('*.py') if '__pycache__' not in x.parts}
    if actual!=expected:raise ValueError('unregistered Python source in bundled diffusers')
    return dict(commit=PINS['code']['commit'],checked_files=len(registered),reconstructed_models_commit=reconstructed.get('commit'))

def verify_weights(root,full=True):
    result=[]
    for row in PINS['weights']['files']:
        p=safe_relative(root,row['path'])
        if not p.is_file() or p.stat().st_size!=row['bytes']:raise ValueError('weight/config absent or wrong size: '+row['path'])
        if full:
            if row.get('sha256') and sha(p)!=row['sha256']:raise ValueError('weight SHA mismatch: '+row['path'])
            if not row.get('sha256') and blob(p)!=row['git_blob_sha1']:raise ValueError('weight metadata blob mismatch')
        result.append(row['path'])
    return dict(revision=PINS['weights']['revision'],files=len(result),full_hashes=full)

def verify_runtime_complete(root):
    import importlib.machinery
    package=Path(root)/'DriveLaW-Video/Infer/diffusers/src/diffusers'
    if importlib.machinery.PathFinder.find_spec('models',[str(package)]) is None:
        raise ValueError('official bundled diffusers/models is absent from pinned source; coordinator must approve a complete official source before install/GPU')
    if not PINS['code'].get('reconstructed_models',{}).get('selection_condition_checked'):
        raise ValueError('reconstructed source selection condition has not passed')

def decode(path,wh):
    import numpy as np
    data=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-xerror','-threads','2','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','-'])
    n=wh[0]*wh[1]*3
    if len(data)%n:raise ValueError('decoded shape mismatch')
    return np.frombuffer(data,dtype=np.uint8).reshape(-1,wh[1],wh[0],3)

def write_video(frames,path,fps):
    """No-clobber lossless writer; a failed partial is retained for diagnosis."""
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    partial=path.with_name(path.stem+'.partial.mp4')
    if partial.exists():raise FileExistsError(partial)
    h,w=frames[0].shape[:2]
    proc=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}',
                           '-r',str(fps),'-i','-','-c:v','libx264rgb','-crf','0','-preset','medium','-threads','2',
                           '-pix_fmt','rgb24','-video_track_timescale','10240','-map_metadata','-1',str(partial)],stdin=subprocess.PIPE)
    try:
        for frame in frames:proc.stdin.write(np.ascontiguousarray(frame,dtype=np.uint8).tobytes())
        proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError('lossless writer failed; partial retained')
        os.link(partial,path);partial.unlink()
    except BaseException:
        if proc.poll() is None:proc.terminate();proc.wait()
        raise

def canvas(native):
    import cv2
    import numpy as np
    if len(native)!=65:raise ValueError('registered native clip must contain65frames')
    return np.stack([cv2.resize(native[native_index(k)],(512,288),interpolation=cv2.INTER_AREA) for k in range(-20,61)])
