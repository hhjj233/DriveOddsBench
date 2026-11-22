#!/usr/bin/env python3
"""Two unchanged official DriveLaW inference calls; keep all native pass outputs."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True
import drivelaw_common as dc

def validate_spec(spec,bench,input_root,frozen_file=None):
    if spec.get('schema')!='wm-calib-bench.drivelaw_video.spec.v1' or spec.get('model')!='drivelaw_video':raise ValueError('DriveLaW spec required')
    split=spec['split'];mode=spec['mode']
    if split=='test':
        expected=dc.B['test_generation_freeze'] if mode=='continue' else dc.B['test_measurement_freeze']
        if not frozen_file or Path(frozen_file).name!=expected:raise ValueError('correct TEST generation/measurement record required')
        dc.verify_freeze(frozen_file,required=[Path(__file__),dc.PINS_PATH])
    dc.guard_names(bench,split,[spec['window']])
    if mode not in ('continue','recon','prefix'):raise ValueError('unregistered mode')
    if spec['settings']!=dc.settings() or spec['generator']['pins_sha256']!=dc.sha(dc.PINS_PATH):raise ValueError('spec settings/pins changed')
    for key,name in [('common_sha256','drivelaw_common.py'),('spec_builder_sha256','make_drivelaw_specs.py'),('runner_sha256','run_drivelaw_bench.py')]:
        if spec['generator'].get(key)!=dc.sha(dc.HERE/name):raise ValueError('spec code provenance changed')
    seed=spec['seed']
    if spec['pass_seeds']!=[seed,seed+1000003]:raise ValueError('pass seed rule changed')
    if dc.hashlib.sha256(spec['prompt'].encode()).hexdigest()!=spec['prompt_sha256']:raise ValueError('prompt hash mismatch')
    for row in spec['inputs'].values():
        p=dc.safe_relative(input_root,row['path'])
        if dc.sha(p)!=row['sha256']:raise ValueError('input hash mismatch')
    motion=json.loads(dc.safe_relative(input_root,spec['inputs']['history_motion.json']['path']).read_text())
    if dc.caption(motion)!=spec['prompt']:raise ValueError('prompt no longer matches past-only pose record')
    if mode=='recon' and not dc.B['reference_policy_approved']:raise ValueError('reference codec policy requires coordinator review before GPU')
    return spec

def load_official(src,weights):
    dc.verify_source(src);dc.verify_runtime_complete(src);dc.verify_weights(weights,full=True)
    import importlib.metadata
    import re
    for package,version in re.findall(r'^([A-Za-z0-9_.-]+)==([^\s\\]+)',(dc.HERE/'requirements-drivelaw-video.lock.txt').read_text(),re.M):
        if importlib.metadata.version(package)!=version:raise ValueError('inference environment differs from exact lock: '+package)
    inference=Path(src)/'DriveLaW-Video/Infer/infer.py'
    # The unchanged entrypoint inserts its own pinned bundled diffusers path.
    if 'diffusers' in sys.modules:raise RuntimeError('clean inference process required; diffusers already imported')
    s=importlib.util.spec_from_file_location('drivelaw_official_infer',inference);module=importlib.util.module_from_spec(s);s.loader.exec_module(module)
    engine=module.LTXInferenceEngine(model_path=str(weights),device='cuda')
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry
    backend,_=_AttentionBackendRegistry.get_active_backend()
    if backend.value!='native':raise ValueError('registered default native SDPA backend required')
    # A permissive from_pretrained warning must never hide an architecture mismatch.
    from safetensors.torch import load_file
    strict=[]
    for component in ('transformer','vae'):
        state={}
        for row in dc.PINS['weights']['files']:
            if row['path'].startswith(component+'/') and row['path'].endswith('.safetensors'):
                shard=load_file(str(Path(weights)/row['path']),device='cpu')
                if set(state)&set(shard):raise ValueError('duplicate state-dict shard keys')
                state.update(shard)
        if not state:raise ValueError('component state dict absent')
        loaded=getattr(engine.pipe,component).load_state_dict(state,strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:raise ValueError('strict state-dict compatibility failed')
        strict.append(dict(component=component,keys=len(state),missing_keys=[],unexpected_keys=[]))
        del state,shard
    engine._bench_strict_loads=strict
    import inspect
    if inspect.signature(engine.pipe.__call__).parameters['frame_rate'].default!=dc.B['temporal_frame_rate']:raise ValueError('official pipeline temporal default changed')
    return module,engine

def vae_roundtrip(engine,frames,seed):
    """Proposed reference/prefix codec path: exact official encode/normalise/decode primitives."""
    import torch
    from diffusers.pipelines.ltx.pipeline_ltx_condition import retrieve_latents
    from diffusers.utils.torch_utils import randn_tensor
    from PIL import Image
    pipe=engine.pipe;generator=torch.Generator(device='cuda').manual_seed(seed)
    images=[Image.fromarray(x) for x in frames]
    tensor=pipe.video_processor.preprocess_video(images,704,1280).to(device='cuda',dtype=pipe.vae.dtype)
    z=retrieve_latents(pipe.vae.encode(tensor),generator=generator)
    z=pipe._normalize_latents(z,pipe.vae.latents_mean,pipe.vae.latents_std).to(device='cuda',dtype=torch.bfloat16)
    z=pipe._denormalize_latents(z,pipe.vae.latents_mean,pipe.vae.latents_std,pipe.vae.config.scaling_factor)
    timestep=None
    if pipe.vae.config.timestep_conditioning:
        noise=randn_tensor(z.shape,generator=generator,device=z.device,dtype=z.dtype)
        timestep=torch.tensor([0.],device=z.device,dtype=z.dtype)
        z=(1-0.)*z+0.*noise
    decoded=pipe.vae.decode(z,timestep,return_dict=False)[0]
    value=pipe.video_processor.postprocess_video(decoded,output_type='pil')[0]
    import numpy as np
    return np.stack([np.asarray(x,dtype=np.uint8) for x in value])

def official_pass(engine,context,prompt,seed):
    from PIL import Image
    import numpy as np
    import torch
    if len(context)!=9:raise ValueError('exact9conditioningframes required')
    # No frame_rate override: preserve the released standalone behaviour.
    torch.cuda.synchronize();started=time.monotonic()
    result=engine.generate_video([Image.fromarray(x) for x in context],prompt,704,1280,33,
                                num_inference_steps=30,seed=seed,noise_reinjection_enabled=True,
                                noise_reinjection_beta=1.,noise_reinjection_sigma=.3,noise_reinjection_steps=3)
    torch.cuda.synchronize()
    engine._bench_pass_timings.append(dict(seed=seed,seconds=time.monotonic()-started,frames=33))
    output=np.stack([np.asarray(x,dtype=np.uint8) for x in result])
    if output.shape!=(33,704,1280,3):raise ValueError('official pass output shape mismatch')
    return output

def continuation_passes(engine,context,prompt,seeds,save_first):
    if len(context)!=9 or len(seeds)!=2 or seeds[1]!=seeds[0]+1000003:
        raise ValueError('registered two-pass context/seed rule required')
    p1=official_pass(engine,context,prompt,seeds[0])
    save_first(p1)
    p2=official_pass(engine,p1[-9:],prompt,seeds[1])
    return p1,p2

def execute_one(module,engine,spec,input_root,output_root,specfile):
    import numpy as np
    import torch
    dest=dc.safe_relative(output_root,spec['output_dir'])
    if dest.exists():raise FileExistsError('generated sample already exists; no overwrite or automatic retry')
    dest.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats();engine._bench_pass_timings=[]
    history=dc.decode(dc.safe_relative(input_root,spec['inputs']['history8_1280.mp4']['path']),(1280,704))
    condition=dc.decode(dc.safe_relative(input_root,spec['inputs']['condition8_1280.mp4']['path']),(1280,704))
    if len(history)!=17 or len(condition)!=9 or not np.array_equal(history[-9:],condition):raise ValueError('conditioning must equal last9realhistoryframes')
    # Prefix is a fixed seeded codec pass; the model itself only reads condition.
    prefix_seed=int.from_bytes(dc.hashlib.sha256(f"bench-v1|{spec['window']}|prefix".encode()).digest()[:4],'big')&0x7fffffff
    with torch.inference_mode():
        prefix=vae_roundtrip(engine,history,prefix_seed)
        if len(prefix)!=17:raise ValueError('prefix codec frame count mismatch')
        dc.write_video(prefix,dest/'prefix-native8.mp4',8)
        if spec['mode']=='prefix':
            native=prefix
            dc.new_json(dest/'sample_outputs.json',dict(schema='wm-calib-bench.drivelaw_video.output.v1',status='success',mode='prefix',run=spec['run'],name=spec['name'],window=spec['window'],split=spec['split'],spec_sha256=dc.sha(specfile),frames=17,native_sha256=dc.sha(dest/'prefix-native8.mp4')))
            return
        if spec['mode']=='continue':
            p1,p2=continuation_passes(engine,condition,spec['prompt'],spec['pass_seeds'],
                lambda frames:dc.write_video(frames,dest/'pass1-native8.mp4',8))
        else:
            real=dc.decode(dc.safe_relative(input_root,spec['inputs']['real8_1280.mp4']['path']),(1280,704))
            if len(real)!=65:raise ValueError('reference needs65raw8Hzframes')
            p1=vae_roundtrip(engine,real[8:41],spec['pass_seeds'][0]);dc.write_video(p1,dest/'pass1-native8.mp4',8)
            p2=vae_roundtrip(engine,real[32:65],spec['pass_seeds'][1])
        dc.write_video(p2,dest/'pass2-native8.mp4',8)
    if p1.shape!=(33,704,1280,3) or p2.shape!=p1.shape:raise ValueError('two33framepasses required')
    native=np.concatenate([prefix[:8],p1,p2[9:]],axis=0)
    dc.write_video(native,dest/'native8.mp4',8);canvas=dc.canvas(native);dc.write_video(canvas,dest/'vision.mp4',10)
    torch.cuda.synchronize();gpu=torch.cuda.get_device_properties(0)
    files={name:dict(sha256=dc.sha(dest/name),bytes=(dest/name).stat().st_size) for name in ['prefix-native8.mp4','pass1-native8.mp4','pass2-native8.mp4','native8.mp4','vision.mp4']}
    record=dict(schema='wm-calib-bench.drivelaw_video.output.v1',status='success',mode=spec['mode'],run=spec['run'],name=spec['name'],window=spec['window'],split=spec['split'],sample=spec['sample'],seed=spec['seed'],pass_seeds=spec['pass_seeds'],spec_sha256=dc.sha(specfile),prompt_sha256=spec['prompt_sha256'],settings=spec['settings'],inputs=spec['inputs'],code_commit=dc.PINS['code']['commit'],weight_revision=dc.PINS['weights']['revision'],files=files,frames=81,native_frames=65,prefix_rule='per_clip; nativefirst8 fromfixedcodec prefix, remaininghistory frompass1decodedconditions',mapping=dc.timeline(),seconds=time.monotonic()-started,gpu_memory=dict(peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20),gpu=dict(name=gpu.name,total_MiB=gpu.total_memory/2**20,device=str(torch.cuda.current_device())),software=dict(torch=torch.__version__,diffusers=module.sys.modules['diffusers'].__version__),attention_backend=dict(torch_SDPA_flash=torch.backends.cuda.flash_sdp_enabled(),torch_SDPA_mem_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),torch_SDPA_math=torch.backends.cuda.math_sdp_enabled(),note='Enabled backend flags; actual processor classes are recorded separately'),attention_processors=sorted({type(p).__module__+'.'+type(p).__name__ for p in engine.pipe.transformer.attn_processors.values()}),pass_wall_times=engine._bench_pass_timings,strict_state_dict_loads=engine._bench_strict_loads,video_sha256=files['vision.mp4']['sha256'])
    dc.new_json(dest/'sample_outputs.json',record)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--specs',type=Path,nargs='+',required=True);ap.add_argument('--bench',type=Path,required=True);ap.add_argument('--input-root',type=Path,required=True);ap.add_argument('--output-root',type=Path,required=True);ap.add_argument('--src',type=Path,required=True);ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--frozen-file',type=Path);ap.add_argument('--dry-run',action='store_true');a=ap.parse_args()
    specs=[]
    for path in a.specs:
        spec=json.loads(path.read_text());validate_spec(spec,a.bench,a.input_root,a.frozen_file);specs.append((path,spec))
    dc.verify_source(a.src)
    if a.dry_run:
        try:dc.verify_runtime_complete(a.src);runtime_ready=True
        except ValueError:runtime_ready=False
        print(json.dumps(dict(CPU_spec_and_source_hashes='PASS',specs=len(specs),source_complete=runtime_ready,GPU_started=False)));return
    dc.verify_runtime_complete(a.src)
    if not dc.B['reference_policy_approved']:raise ValueError('codec prefix/reference policy not approved in drivelaw_pins.json; no GPU started')
    if os.environ.get('CUDA_VISIBLE_DEVICES') is None:raise ValueError('launch through approved GPU wrapper')
    if not (a.weights/'download-verified.json').is_file():raise ValueError('owner-approved pinned weight download proof required')
    module,engine=load_official(a.src,a.weights)
    for path,spec in specs:
        try:execute_one(module,engine,spec,a.input_root,a.output_root,path)
        except BaseException as error:
            dest=dc.safe_relative(a.output_root,spec['output_dir'])
            if dest.is_dir() and not (dest/'failure.json').exists():dc.new_json(dest/'failure.json',dict(error_type=type(error).__name__,failed=True,no_retry=True,all_partial_outputs_retained=True))
            raise
        print(json.dumps(dict(name=spec['name'],status='success')),flush=True)
if __name__=='__main__':main()
