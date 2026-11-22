#!/usr/bin/env python3
"""Check DEV outputs and retain scalar diagnostics; never opens TEST samples."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
import numpy as np
import drivelaw_common as dc

def check(sample):
    record=json.loads((sample/'sample_outputs.json').read_text())
    if record.get('split')!='dev':raise ValueError('DEV smoke only; no TEST decoding')
    if record.get('status')!='success' or record.get('mode') not in ('continue','recon'):raise ValueError('completed continuation/reference required')
    if record['settings']!=dc.settings():raise ValueError('settings drift')
    if record['pass_seeds']!=[record['seed'],record['seed']+1000003]:raise ValueError('seed rule drift')
    clips={}
    for name,row in record['files'].items():
        path=dc.safe_relative(sample,name)
        if dc.sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('output hash mismatch')
        meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(path)]))['streams'][0]
        wh=(512,288) if name=='vision.mp4' else (1280,704);fps=10 if name=='vision.mp4' else 8
        if (meta['width'],meta['height'])!=wh or meta['r_frame_rate']!=str(fps)+'/1':raise ValueError('size/fps mismatch')
        clips[name]=dc.decode(path,wh)
    prefix,p1,p2,native,vision=[clips[x] for x in ['prefix-native8.mp4','pass1-native8.mp4','pass2-native8.mp4','native8.mp4','vision.mp4']]
    if [len(x) for x in [prefix,p1,p2,native,vision]]!=[17,33,33,65,81]:raise ValueError('frame count mismatch')
    if not np.array_equal(native,np.concatenate([prefix[:8],p1,p2[9:]])):raise ValueError('native assembly mismatch')
    if not np.array_equal(vision,dc.canvas(native)):raise ValueError('8->10 mapping mismatch')
    if len(record.get('strict_state_dict_loads',[]))!=2 or any(x['missing_keys'] or x['unexpected_keys'] for x in record['strict_state_dict_loads']):raise ValueError('strict checkpoint loading evidence absent')
    def jump(i):return float(np.abs(vision[i].astype(float)-vision[i-1]).mean())
    return dict(window=record['window'],mode=record['mode'],sample=record['sample'],canvas_sha256=record['files']['vision.mp4']['sha256'],format_hashes_assembly_mapping='PASS',
                t0_to_first_future_canvas_mean_RGB=jump(21),pass_seam_canvas_mean_RGB=jump(51),
                seconds=record['seconds'],gpu_memory=record['gpu_memory'],gpu=record['gpu'],attention_backend=record['attention_backend'],attention_processors=record['attention_processors'])

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--samples',type=Path,nargs='+',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
    results=[check(p) for p in a.samples];windows={x['window'] for x in results}
    if len(windows)!=2 or len(results)!=6:raise ValueError('smoke needs exactly2DEVwindows x2continuations plus2references')
    for w in windows:
        rows=[x for x in results if x['window']==w]
        if {x['sample'] for x in rows if x['mode']=='continue'}!={0,1} or sum(x['mode']=='recon' for x in rows)!=1:raise ValueError('smoke sample set differs')
        if len({x['canvas_sha256'] for x in rows if x['mode']=='continue'})!=2:raise ValueError('seed diversity failed: both continuation clips byte-identical')
    dc.new_json(a.report,dict(structural_checks='PASS',results=results,visual_continuation_and_A1_pending=True,diagnostics_are_not_quality_gates=True))
    print(json.dumps(dict(structural_checks='PASS',samples=len(results),visual_continuation_and_A1_pending=True)))
if __name__=='__main__':main()
