#!/usr/bin/env python3
"""AST-only check of every direct VAE/transformer/pipeline call in shipped inference."""
import argparse
import ast
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import drivelaw_common as dc

def audit(src):
    src=Path(src);dc.verify_source(src);dc.verify_runtime_complete(src)
    package=src/'DriveLaW-Video/Infer/diffusers/src/diffusers'
    classes={};trees={}
    for f in package.rglob('*.py'):
        tree=ast.parse(f.read_text());trees[f]=tree
        for n in tree.body:
            if isinstance(n,ast.ClassDef):classes[n.name]=(f,n)
    def method(cls,name,seen=()):
        if cls in seen or cls not in classes:return None
        f,node=classes[cls]
        own=next((n for n in node.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name),None)
        if own:return f,own
        for base in node.bases:
            got=method(ast.unparse(base),name,(*seen,cls))
            if got:return got
    checked=[];sources=[src/'DriveLaW-Video/Infer/infer.py',package/'pipelines/ltx/pipeline_ltx_condition.py']
    for f in sources:
        tree=ast.parse(f.read_text())
        for call in ast.walk(tree):
            if not isinstance(call,ast.Call):continue
            expr=ast.unparse(call.func);cls=None;name=None
            if expr=='self.transformer':cls,name='LTXVideoTransformer3DModel','forward'
            elif expr=='self.pipe':cls,name='LTXConditionPipeline','__call__'
            elif expr.startswith('self.vae.') and expr.count('.')==2:cls,name='AutoencoderKLLTXVideo',expr.split('.')[-1]
            elif expr=='self.pipe.vae.enable_tiling':cls,name='AutoencoderKLLTXVideo','enable_tiling'
            elif expr=='LTXConditionPipeline.from_pretrained':cls,name='LTXConditionPipeline','from_pretrained'
            if cls is None:continue
            resolved=method(cls,name)
            if not resolved:raise ValueError('required method absent: '+cls+'.'+name)
            target,definition=resolved;args=definition.args
            accepted={x.arg for x in [*args.posonlyargs,*args.args,*args.kwonlyargs]}
            unknown={k.arg for k in call.keywords if k.arg is not None}-accepted
            if unknown and args.kwarg is None:raise ValueError('unsupported keywords '+str(unknown)+' in '+expr)
            positional=[x for x in [*args.posonlyargs,*args.args] if x.arg not in ('self','cls')]
            if len(call.args)>len(positional) and args.vararg is None:raise ValueError('too many positional args: '+expr)
            checked.append(dict(source=str(f.relative_to(src)),line=call.lineno,call=expr,keywords=[k.arg for k in call.keywords],class_name=cls,method=name,definition=str(target.relative_to(src)),definition_line=definition.lineno,signature=ast.unparse(args)))
    required={('LTXVideoTransformer3DModel','forward'),('AutoencoderKLLTXVideo','encode'),('AutoencoderKLLTXVideo','decode'),('AutoencoderKLLTXVideo','enable_tiling')}
    if not required<={(x['class_name'],x['method']) for x in checked}:raise ValueError('incomplete API coverage')
    return dict(static_direct_call_check='PASS',calls=checked,code_hashes=dc.verify_source(src),strict_checkpoint_loads_pending=True,dynamic_import_and_GPU_compatibility_not_tested=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--src',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
    report=audit(a.src);dc.new_json(a.report,report);print(json.dumps(dict(static_direct_call_check='PASS',calls=len(report['calls']),GPU_started=False)))
if __name__=='__main__':main()
