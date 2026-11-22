#!/usr/bin/env python3
"""CPU-only architecture and state-dict check, run before any GPU job."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True
import drivelaw_common as dc

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--src',type=Path,required=True);ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
    if a.report.exists():raise FileExistsError('fresh check report required')
    dc.verify_source(a.src);dc.verify_runtime_complete(a.src);dc.verify_weights(a.weights)
    if 'diffusers' in sys.modules:raise RuntimeError('fresh process required')
    sys.path.insert(0,str(a.src/'DriveLaW-Video/Infer/diffusers/src'))
    import torch
    from safetensors.torch import load_file
    from diffusers import AutoencoderKLLTXVideo,LTXVideoTransformer3DModel
    rows=[]
    for name,cls in [('transformer',LTXVideoTransformer3DModel),('vae',AutoencoderKLLTXVideo)]:
        started=time.monotonic()
        with torch.device('meta'):model=cls.from_config(json.loads((a.weights/name/'config.json').read_text()))
        state={}
        for row in dc.PINS['weights']['files']:
            if row['path'].startswith(name+'/') and row['path'].endswith('.safetensors'):
                shard=load_file(str(a.weights/row['path']),device='cpu')
                if set(state)&set(shard):raise ValueError('duplicate state-dict keys')
                state.update(shard)
        if not state:raise ValueError('empty checkpoint')
        loaded=model.load_state_dict(state,strict=True,assign=True)
        if loaded.missing_keys or loaded.unexpected_keys:raise ValueError('strict load mismatch')
        rows.append(dict(component=name,keys=len(state),missing_keys=[],unexpected_keys=[],seconds=time.monotonic()-started,CPU_only=True))
        del state,shard,model
    dc.new_json(a.report,dict(PASS=True,strict=True,components=rows,GPU_started=False,pins_sha256=dc.sha(dc.PINS_PATH),torch=torch.__version__))
    print(json.dumps(dict(PASS=True,components=len(rows),GPU_started=False)))
if __name__=='__main__':main()
