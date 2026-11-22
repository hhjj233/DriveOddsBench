#!/usr/bin/env python3
"""Pinned K=16 DriveLaW specs; history-grounded prompt fixed across both passes."""
from pathlib import Path
import argparse
import json
import sys
sys.dont_write_bytecode=True
import drivelaw_common as dc

def parse_samples(text):
    result=[]
    for item in text.split(','):
        lo,_,hi=item.partition('-');result.extend(range(int(lo),int(hi or lo)+1))
    if len(result)!=len(set(result)) or any(k not in range(16) for k in result):raise ValueError('unique samples0..15 required')
    return result

def build(bench,input_root,split,windows,mode,run,samples,snapshot,out,frozen_file=None):
    dc.guard_names(bench,split,windows);settings=dc.settings()
    if not run.startswith('drivelaw-'+split+'-') or not mode in ('continue','recon','prefix'):raise ValueError('registered run/split/mode required')
    if split=='test' and mode!='continue':dc.verify_freeze(frozen_file or '',required=[dc.PINS_PATH,Path(__file__)])
    written=[]
    for w in windows:
        root=Path(input_root)/'windows'/split/w;record=json.loads((root/'inputs.json').read_text())
        if record['window']!=w or record['split']!=split:raise ValueError('input split mismatch')
        if record['pins_sha256']!=dc.sha(dc.PINS_PATH) or record['generator_sha256']!=dc.sha(dc.HERE/'build_drivelaw_inputs.py') or record['common_sha256']!=dc.sha(dc.HERE/'drivelaw_common.py'):raise ValueError('input build provenance changed; prepare a new candidate input directory')
        motion=json.loads((root/'history_motion.json').read_text())
        prompt=dc.caption(motion)
        needed=['history8_1280.mp4','condition8_1280.mp4','history_motion.json']
        if mode=='recon':needed.append('real8_1280.mp4')
        inputs={}
        for name in needed:
            path=root/name
            if dc.sha(path)!=record['files'][name]['sha256']:raise ValueError('input digest changed')
            inputs[name]=dict(path=str(path.relative_to(input_root)),sha256=dc.sha(path))
        keys=samples if mode=='continue' else [mode]
        for k in keys:
            seed=dc.shared_seed(w,k,snapshot);name=w+('-s'+str(k) if mode=='continue' else '-'+mode)
            spec=dict(schema='wm-calib-bench.drivelaw_video.spec.v1',model='drivelaw_video',split=split,window=w,
                      mode=mode,run=run,name=name,sample=k,seed=seed,pass_seeds=[seed,seed+1000003],
                      inputs=inputs,prompt=prompt,prompt_sha256=dc.hashlib.sha256(prompt.encode()).hexdigest(),
                      prompt_rule='history only,no navigation; pass2prompt verbatim',settings=settings,
                      output_dir=run+'/native/'+name,keep=['native8','pass1','pass2'],
                      generator=dict(pins_sha256=dc.sha(dc.PINS_PATH),spec_builder_sha256=dc.sha(Path(__file__)),common_sha256=dc.sha(dc.HERE/'drivelaw_common.py'),runner_sha256=dc.sha(dc.HERE/'run_drivelaw_bench.py')))
            if split=='test' and mode!='continue':spec['measurement_freeze']=dc.verify_freeze(frozen_file)
            path=Path(out)/run/(name+'.json');dc.new_json(path,spec);written.append(str(path))
    return written

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--bench',type=Path,required=True);ap.add_argument('--input-root',type=Path,required=True);ap.add_argument('--split',choices=['dev','test'],required=True);ap.add_argument('--names',type=Path,required=True);ap.add_argument('--mode',choices=['continue','recon','prefix'],required=True);ap.add_argument('--run',required=True);ap.add_argument('--samples',default='0-15');ap.add_argument('--snapshot',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--frozen-file',type=Path);a=ap.parse_args()
    if a.split=='test' and a.mode!='continue':dc.verify_freeze(a.frozen_file or '',required=[dc.PINS_PATH,Path(__file__)])
    files=build(a.bench,a.input_root,a.split,dc.names(a.names),a.mode,a.run,parse_samples(a.samples),a.snapshot,a.out,a.frozen_file)
    print(json.dumps(dict(specs=len(files),mode=a.mode,split=a.split)))
if __name__=='__main__':main()
