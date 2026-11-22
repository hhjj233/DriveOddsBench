#!/usr/bin/env python3
"""VaViM bench runner: load the released VaViM-L once, then run every spec, one batch per window.

  run_vavim_bench.py --specs SPEC... --bench BENCH --input-root INPUTS --output-root OUT --src SOURCE --weights WEIGHTS
                     [--frozen-file RECORD] [--equivalence-check [--equivalence-frames N]] [--dry-run]

Per window: the 21 history frames (ticks -20..0) are tokenised in ONE call of the released LlamaGen encoder and decoded in
ONE call of the released detokenizer (the shared codec prefix; deterministic VQ). The context tokens are the encoded rows
at the context ticks (default -20, -15, -10, -5, 0 = 2 Hz over the 2.0 s history). The window's continuation specs run as
ONE batch through the released MupGPT2.forward_inference (12 future frames = ticks 5, 10, ..., 60; KV cache; the released
rolling context of 7 frames), with one change: each batch row samples from its own torch.Generator (the instance's
_sample_next_token is replaced by the same arithmetic with a per-row torch.multinomial). Generated tokens are decoded per
sample (one detokenizer call of 12 frames). The reference spec round-trips the real future frames at the same 2 Hz ticks
through the same codec. The 17-frame 2 Hz native clip (5 prefix + 12 future frames) is mapped to the 81-frame 10 Hz canvas
by the registered rule (vavim_common.timeline). Outputs, never overwritten:
<OUT>/<run>/native/<window>-s<k>/ (or -recon/): vision.mp4 (81 x 512x288, 10 fps, lossless), native2.mp4 (17 x 512x288,
2 fps, lossless), tokens.npz (history, context and generated or reference token grids), sample_outputs.json.
A failure leaves failure.json and every partial file; no automatic retry.
--equivalence-check (first window): one batch-1 sample (seed_of(window, "equivalence")) through the UNCHANGED released
forward_inference after torch.manual_seed(seed), and through the kit path with a per-row generator seeded alike; the
token grids must be identical (<OUT>/<run>/equivalence-check-<window>.json).
I/O: every path argument must be absolute; the runner writes only under --output-root and never uses the working directory.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import contextlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import types

sys.dont_write_bytecode = True
import vavim_common as vc  # noqa: E402
import make_vavim_specs as ms  # noqa: E402

DEVIATIONS = [
    'per-sample torch.Generator: the instance attribute _sample_next_token replaces the released method with the same '
    'arithmetic (softmax at the temperature, top-k sorted=False, renormalise, multinomial, gather) and one '
    'torch.multinomial per batch row from that row\'s generator (released: one multinomial over the batch from the global '
    'RNG); proven token-identical to the released function at batch 1 by --equivalence-check',
    'stochastic sampling (registered top_k, temperature 1.0) instead of the released default topk_sampler=1 (greedy), so that '
    'K continuations differ',
    'the mup base-shape file named in the checkpoint (relative path ./mup_shapes/gpt2_24layers_basewidth256.bsh, resolved '
    'against the working directory by the released load_pretrained_gpt) is resolved against the verified source checkout',
    'checkpoint pickle globals scanned by opcode against an allow-list before the released torch.load(map_location=cpu)',
    'model frames are the bench canvas render (cv2.INTER_AREA of the band) instead of the released torchvision bilinear '
    'antialias resize of 16:9 sources; same 512x288 size',
    'frames written losslessly (libx264rgb crf 0); the uint8 frames are the released torch_image_to_plot conversion',
    'cudnn.deterministic True, cudnn.benchmark False (the released scripts set TF32 on, which has no effect under bf16 '
    'autocast for the GPT; PyTorch TF32 defaults are kept and recorded)',
    'a frame-timing hook wraps the instance\'s _process_generated_frame (synchronises and records a timestamp, then calls '
    'the released method)',
    'each window runs under torch.inference_mode (the released forward_inference uses torch.no_grad; values unchanged)',
]


def validate_spec(spec, bench, input_root, frozen_file=None):
    if spec.get('schema') != 'wm-calib-bench.vavim.spec.v1' or spec.get('model') != 'vavim':
        raise ValueError('VaViM spec required')
    split, mode = spec['split'], spec['mode']
    if mode not in ('continue', 'recon'):
        raise ValueError('unregistered mode')
    if split == 'test':
        expected = vc.B['test_generation_freeze'] if mode == 'continue' else vc.B['test_measurement_freeze']
        if not frozen_file or Path(frozen_file).name != expected:
            raise ValueError('correct TEST generation/measurement record required')
        vc.verify_freeze(frozen_file, required=[Path(__file__), vc.PINS_PATH, vc.HERE / 'vavim_common.py'])
    purpose = spec.get('purpose')
    diagnostic = purpose == 'diagnostic'
    if bool(spec.get('diagnostic')) != diagnostic:
        raise ValueError('diagnostic flag and purpose disagree')
    vc.check_purpose(purpose, split, spec['run'], mode, spec['window'], spec['sample'])
    vc.guard_names(bench, split, [spec['window']], diagnostic)
    listed_name = 'names-dev.txt' if diagnostic else 'names-' + split + '-gen.txt'
    if spec.get('names_list') != dict(file=listed_name, sha256=vc.sha(Path(bench) / listed_name)):
        raise ValueError('spec was built against a different window list')
    if spec['settings'] != vc.settings():
        raise ValueError('spec settings changed')
    if spec['generator'] != ms.kit_hashes():
        raise ValueError('spec code provenance changed')
    if mode == 'continue' and (spec['sample'] not in range(16) or spec['seed'] != vc.local_seed(spec['window'], spec['sample'])):
        raise ValueError('seed rule changed')
    if mode == 'recon' and (spec['seed'] is not None or 'real10.mp4' not in spec['inputs']):
        raise ValueError('reference needs real10.mp4 and no seed')
    for row in spec['inputs'].values():
        if vc.sha(vc.safe_relative(input_root, row['path'])) != row['sha256']:
            raise ValueError('input hash mismatch')
    return spec


def check_environment():
    """Exact lock: every name==version pin, and the mup direct-URL archive."""
    lock = (vc.HERE / 'requirements-vavim.lock.txt').read_text()
    pins = re.findall(r'^([A-Za-z0-9_.-]+)==([^\s\\]+)', lock, re.M)
    for package, version in pins:
        if importlib.metadata.version(package) != version:
            raise ValueError('inference environment differs from the exact lock: ' + package)
    url = re.search(r'^mup @ (\S+)', lock, re.M).group(1)
    direct = json.loads(importlib.metadata.distribution('mup').read_text('direct_url.json') or '{}')
    if direct.get('url') != url:
        raise ValueError('mup is not the pinned archive')
    return len(pins) + 1


def gpu_record(torch):
    if not torch.cuda.is_available():
        return dict(name=None, device='cpu')
    p = torch.cuda.get_device_properties(0)
    rec = dict(name=p.name, total_MiB=p.total_memory / 2**20, capability=f'{p.major}.{p.minor}',
               CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'), torch_cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version(), uuid=str(getattr(p, 'uuid', '')))
    try:
        query = ['nvidia-smi', '--query-gpu=uuid,name,driver_version,pci.bus_id', '--format=csv,noheader']
        visible = (os.environ.get('CUDA_VISIBLE_DEVICES') or '').split(',')[0].strip()
        rec['nvidia_smi_device0'] = subprocess.run(query + (['--id=' + visible] if visible else []), capture_output=True,
                                                   text=True, timeout=30).stdout.strip().splitlines() if visible else None
    except (OSError, subprocess.SubprocessError):
        rec['nvidia_smi_device0'] = None
    return rec


def software(torch):
    out = dict(torch=torch.__version__, cuda=torch.version.cuda, python=sys.version.split()[0], python_executable=sys.executable,
               env_prefix=sys.prefix)
    for name in ('mup', 'lightning', 'hydra-core', 'omegaconf', 'einops', 'numpy'):
        out[name] = importlib.metadata.version(name)
    return out


def backend_record(torch):
    return dict(sdpa_flash=torch.backends.cuda.flash_sdp_enabled(), sdpa_mem_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                sdpa_math=torch.backends.cuda.math_sdp_enabled(), cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                note='the released attention passes a boolean causal mask (prefill) or none (KV-cache steps) to '
                     'scaled_dot_product_attention; the kernel chosen per call is not profiled')


def load_official(src, weights, device='cuda'):
    """The released load_pretrained_gpt sequence + strict checks; the released TorchScript tokenizer and detokenizer."""
    src = Path(src).resolve()
    vc.verify_source(src)
    vc.verify_weights(weights, full=True)
    globals_used = {role: vc.check_pickle_globals(vc.weight_path(weights, role), role if role != 'gpt_checkpoint' else 'checkpoint')
                    for role in ('gpt_checkpoint', 'tokenizer_encoder', 'tokenizer_decoder')}
    locked = check_environment()
    if any(m == 'vam' or m.startswith('vam.') for m in sys.modules):
        raise RuntimeError('clean process required')
    sys.path.insert(0, str(src))
    import torch
    import mup
    from vam.video_pretraining import mup_gpt2
    if Path(mup_gpt2.__file__).resolve() != src / 'vam/video_pretraining/mup_gpt2.py':
        raise RuntimeError('official module imported from the wrong place')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(device)
    t0 = time.monotonic()
    # ---- released load_pretrained_gpt (vam/video_pretraining/mup_gpt2.py), same calls and order ----
    ckpt = torch.load(str(vc.weight_path(weights, 'gpt_checkpoint')), map_location='cpu', weights_only=False)
    config = ckpt['hyper_parameters']['network'].copy()
    config.pop('_target_')
    want = vc.B['gpt_config']
    if dict(config) != want:
        raise ValueError('checkpoint network config differs from the pins: ' + json.dumps(dict(config)))
    gpt = mup_gpt2.MupGPT2(**config)
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('network.', '')] = v
    own = gpt.state_dict()
    missing, unexpected = sorted(set(own) - set(state_dict)), sorted(set(state_dict) - set(own))
    shapes = [k for k in own if k in state_dict and tuple(own[k].shape) != tuple(state_dict[k].shape)]
    if missing or unexpected or shapes:
        raise ValueError(f'state dict mismatch: missing {missing[:5]} unexpected {unexpected[:5]} shapes {shapes[:5]}')
    loaded = gpt.load_state_dict(state_dict)                     # released call (strict=True by default)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError('strict load failed')
    base = ckpt['hyper_parameters']['mup_base_shapes']
    if base != './' + vc.PINS['code']['mup_base_shapes_file']:
        raise ValueError('unexpected mup base-shape reference: ' + repr(base))
    base_path = src / vc.PINS['code']['mup_base_shapes_file']     # deviation: resolved against the verified checkout
    mup.set_base_shapes(gpt, str(base_path), rescale_params=False)
    _ = gpt.eval()
    _ = gpt.to(device)
    _ = gpt.requires_grad_(False)
    # ---- checks of the loaded model ----
    yaml_base = {n: [d.base_dim for d in v] for n, v in mup.shape.load_base_shapes(str(base_path)).items()}
    saved_base = {n: [d.base_dim for d in s] for n, s in (ckpt.get('mup_base_shapes') or {}).items()}
    model_base = {n: [d.base_dim for d in p.infshape] for n, p in gpt.named_parameters()}
    base_agree = yaml_base == saved_base == model_base
    if not base_agree:
        raise ValueError('mup base shapes of the file, the checkpoint record and the model disagree')
    width_mult = float(gpt.lm_head.width_mult())
    if width_mult != vc.B['lm_head_width_mult'] or gpt.lm_head.weight is not gpt.transformer.wie.weight:
        raise ValueError('readout scaling or weight tying differs from the training setup')
    n_unique = sum(p.numel() for p in gpt.parameters())
    record = dict(strict_state_dict_load=dict(keys=len(own), checkpoint_tensors=len(state_dict), missing_keys=[], unexpected_keys=[],
                                              shape_mismatches=[], parameters_unique=n_unique,
                                              parameters_in_state_dict=sum(v.numel() for v in state_dict.values()),
                                              checkpoint_dtypes=sorted({str(v.dtype) for v in state_dict.values()}),
                                              loader='released MupGPT2(**hyper_parameters.network) + load_state_dict (strict)'),
                  network_config=dict(config), global_step=ckpt.get('global_step'), epoch=ckpt.get('epoch'),
                  pytorch_lightning_version=ckpt.get('pytorch-lightning_version'), mup_base_shapes=dict(
                      checkpoint_reference=base, resolved_file=vc.PINS['code']['mup_base_shapes_file'],
                      file_git_blob_sha1=vc.blob(base_path), agrees_with_saved_infshapes=bool(base_agree),
                      lm_head_width_mult=width_mult, output_mult=float(gpt.lm_head.output_mult)),
                  pickle_globals=globals_used, locked_packages=locked)
    del ckpt, state_dict
    encoder = torch.jit.load(str(vc.weight_path(weights, 'tokenizer_encoder')), map_location=device)
    decoder = torch.jit.load(str(vc.weight_path(weights, 'tokenizer_decoder')), map_location=device)
    record['tokenizer'] = dict(encoder_class=encoder.original_name, decoder_class=decoder.original_name,
                               frozen_graphs='traced and frozen TorchScript (parameters are graph constants; no training flag)',
                               loader='torch.jit.load(path, map_location=device): the frozen TorchScript constants load on the device, as the released torch.jit.load(path).to(cuda) (loading on cpu first leaves them on the cpu)')
    record['load_seconds'] = time.monotonic() - t0
    return types.SimpleNamespace(torch=torch, gpt=gpt, encoder=encoder, decoder=decoder, device=device, record=record,
                                 mup_gpt2=mup_gpt2)


def autocast(engine, part):
    """Registered precision per part (pins bench.autocast): bfloat16 autocast on the engine's device type."""
    rule = vc.B['autocast'][part]
    if not rule.startswith('torch.autocast(cuda, bfloat16)'):
        return contextlib.nullcontext()
    return engine.torch.autocast(device_type=engine.device.type, dtype=engine.torch.bfloat16)


def encode(engine, frames_u8):
    torch = engine.torch
    with autocast(engine, 'encoder'):
        tokens = engine.encoder(vc.to_model_input(torch, frames_u8, engine.device))
    if tuple(tokens.shape) != (len(frames_u8), *vc.TOKEN_HW) or tokens.dtype != torch.int64:
        raise ValueError('encoder output shape/dtype changed: ' + str(tuple(tokens.shape)))
    vc.check_tokens(tokens.cpu().numpy())
    return tokens


def decode_tokens(engine, tokens):
    torch = engine.torch
    vc.check_tokens(tokens.cpu().numpy())
    with autocast(engine, 'decoder'):
        images = engine.decoder(tokens)
    if tuple(images.shape) != (len(tokens), 3, vc.MODEL_WH[1], vc.MODEL_WH[0]):
        raise ValueError('decoder output shape changed')
    return vc.to_uint8(torch, images)


def per_row_sampler(torch, generators):
    """MupGPT2._sample_next_token with one torch.multinomial per row from the row's own generator."""
    def sample(logits, temperature, topk_sampler):
        logits = logits[:, -1, :] / temperature
        tokens_probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_tokens = torch.topk(tokens_probs, topk_sampler, dim=-1, sorted=False)
        topk_renormalized_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        if topk_renormalized_probs.shape[0] != len(generators):
            raise RuntimeError('one generator per batch row required')
        next_token_idx = torch.cat([torch.multinomial(topk_renormalized_probs[i:i + 1], num_samples=1, generator=g)
                                    for i, g in enumerate(generators)], dim=0)
        return torch.gather(topk_tokens, -1, next_token_idx)
    return sample


@contextlib.contextmanager
def kit_sampling(engine, generators, stamps=None):
    """Instance-level override of the sampler (and an optional frame-timing hook); the class code is untouched."""
    torch, gpt = engine.torch, engine.gpt
    if '_sample_next_token' in vars(gpt) or '_process_generated_frame' in vars(gpt):
        raise RuntimeError('sampler override already installed')
    gpt._sample_next_token = per_row_sampler(torch, generators)
    if stamps is not None:
        released = type(gpt)._process_generated_frame.__get__(gpt)

        def timed(context, height, width):
            if engine.device.type == 'cuda':
                torch.cuda.synchronize()
            stamps.append(time.monotonic())
            return released(context, height, width)
        gpt._process_generated_frame = timed
    try:
        yield
    finally:
        del gpt._sample_next_token
        if stamps is not None:
            del gpt._process_generated_frame


def generate(engine, burnin, seeds, stamps=None):
    """Kit path: the released forward_inference, batch = len(seeds) copies of the context, per-row generators."""
    torch = engine.torch
    s = vc.B['sampling']
    gens = [torch.Generator(device=engine.device).manual_seed(int(seed)) for seed in seeds]
    batch = burnin.unsqueeze(0).expand(len(seeds), *burnin.shape).contiguous()
    with kit_sampling(engine, gens, stamps), autocast(engine, 'gpt'):
        out = engine.gpt.forward_inference(number_of_future_frames=len(vc.FUTURE_K), burnin_visual_tokens=batch,
                                           temperature=s['temperature'], topk_sampler=s['top_k'], use_kv_cache=s['use_kv_cache'])
    return out


def sync(engine):
    if engine.device.type == 'cuda':
        engine.torch.cuda.synchronize()


def equivalence_check(engine, burnin, window, path, frames):
    """Released forward_inference (global RNG after torch.manual_seed) vs the kit path at batch 1, same seed."""
    torch = engine.torch
    s = vc.B['sampling']
    seed = vc.local_seed(window, 'equivalence')
    b1 = burnin.unsqueeze(0).contiguous()
    sync(engine)
    t1 = time.monotonic()
    torch.manual_seed(seed)
    with autocast(engine, 'gpt'):
        released = engine.gpt.forward_inference(number_of_future_frames=frames, burnin_visual_tokens=b1, temperature=s['temperature'],
                                                topk_sampler=s['top_k'], use_kv_cache=s['use_kv_cache'])
    sync(engine)
    t2 = time.monotonic()
    gens = [torch.Generator(device=engine.device).manual_seed(seed)]
    with kit_sampling(engine, gens), autocast(engine, 'gpt'):
        kit = engine.gpt.forward_inference(number_of_future_frames=frames, burnin_visual_tokens=b1, temperature=s['temperature'],
                                           topk_sampler=s['top_k'], use_kv_cache=s['use_kv_cache'])
    sync(engine)
    t3 = time.monotonic()
    a, b = released.cpu(), kit.cpu()
    same = bool(torch.equal(a, b))
    rec = dict(window=window, seed=seed, seed_rule=vc.B['equivalence_seed_rule'], frames=frames, batch=1, identical_tokens=same,
               differing_tokens=int((a != b).sum()), tokens_sha256=vc.sha_bytes(a.numpy().astype('int16').tobytes()),
               released_seconds=t2 - t1, kit_seconds=t3 - t2, released='MupGPT2.forward_inference after torch.manual_seed(seed)',
               kit='same call with the per-row sampler (torch.Generator(device).manual_seed(seed))', gpu=gpu_record(torch),
               settings=dict(temperature=s['temperature'], top_k=s['top_k'], use_kv_cache=s['use_kv_cache'], autocast=vc.B['autocast']['gpt']))
    vc.new_json(path, rec)
    if not same:
        raise ValueError('kit sampling path differs from the released forward_inference')
    return rec


def write_sample(dest, spec, specfile, native2, prefix10, tokens, record):
    files = {'native2.mp4': vc.write_checked(list(native2), dest / 'native2.mp4', vc.B['model_fps'])}
    vision = vc.assemble_canvas(native2, prefix10)
    files['vision.mp4'] = vc.write_checked(list(vision), dest / 'vision.mp4', vc.B['canvas_fps'])
    import numpy as np
    np.savez_compressed(dest / 'tokens.npz', **{k: np.asarray(v).astype(np.int16) for k, v in tokens.items()})
    files['tokens.npz'] = dict(sha256=vc.sha(dest / 'tokens.npz'), bytes=(dest / 'tokens.npz').stat().st_size)
    record.update(files=files, video_sha256=files['vision.mp4']['sha256'], spec_sha256=vc.sha(specfile))
    vc.new_json(dest / 'sample_outputs.json', record)


def run_window(engine, items, a, common_record, equivalence=False, equivalence_frames=12):
    import numpy as np
    torch = engine.torch
    spec0 = items[0][1]
    window, split = spec0['window'], spec0['split']
    started = time.monotonic()
    if engine.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    hist_u8 = vc.decode(vc.safe_relative(a.input_root, spec0['inputs']['history10.mp4']['path']), vc.MODEL_WH)
    ctx_u8 = vc.decode(vc.safe_relative(a.input_root, spec0['inputs']['context2.mp4']['path']), vc.MODEL_WH)
    ctx_rows = [k + 20 for k in vc.CONTEXT_K]
    if len(hist_u8) != len(vc.HISTORY_K) or not np.array_equal(ctx_u8, hist_u8[ctx_rows]):
        raise ValueError('context frames must be the history frames at the context ticks')
    with torch.inference_mode():
        hist_tokens = encode(engine, hist_u8)                       # one encoder call, 21 frames
        prefix10 = decode_tokens(engine, hist_tokens)               # one detokenizer call, 21 frames
        prefix2 = prefix10[::vc.STEP]                               # ticks -20, -15, -10, -5, 0
        burnin = hist_tokens[ctx_rows]
        sync(engine)
        t_prefix = time.monotonic() - started
        common = dict(common_record, prefix_frames10_sha256=vc.sha_bytes(prefix10.tobytes()),
                      prefix_frames2_sha256=vc.sha_bytes(prefix2.tobytes()),
                      context_tokens_sha256=vc.sha_bytes(burnin.cpu().numpy().astype('int16').tobytes()))
        if equivalence:
            path = vc.safe_relative(a.output_root, spec0['run']) / f'equivalence-check-{window}.json'
            equivalence_check(engine, burnin, window, path, equivalence_frames)
        cont = sorted([x for x in items if x[1]['mode'] == 'continue'], key=lambda x: x[1]['sample'])
        recon = [x for x in items if x[1]['mode'] == 'recon']
        dests = {}
        for _, spec in cont + recon:
            dest = vc.safe_relative(a.output_root, spec['output_dir'])
            if dest.exists():
                raise FileExistsError('generated sample already exists; no overwrite or automatic retry')
            dest.mkdir(parents=True, exist_ok=False)
            dests[spec['name']] = dest
        base = dict(settings=vc.settings(), timeline_rule=dict(canvas_map=vc.B['canvas_map'], prefix_rate=vc.B['prefix_rate']),
                    timeline=vc.timeline(), frames=81, native_frames=len(vc.NATIVE2_K), native2_ticks=vc.NATIVE2_K,
                    prefix_rule=vc.B['prefix_rules'][vc.B['prefix_rate']], prefix_encode=vc.B['prefix_encode'])
        if cont:
            seeds = [s['seed'] for _, s in cont]
            stamps = []
            sync(engine)
            t_gen = time.monotonic()
            gen = generate(engine, burnin, seeds, stamps)
            sync(engine)
            gen_seconds = time.monotonic() - t_gen
            if tuple(gen.shape) != (len(seeds), len(vc.FUTURE_K), *vc.TOKEN_HW):
                raise ValueError('forward_inference output shape changed')
            gen_np = vc.check_tokens(gen.cpu().numpy())
            steps = np.diff([t_gen] + stamps)
            peak = (dict(peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20, peak_reserved_MiB=torch.cuda.max_memory_reserved() / 2**20)
                    if engine.device.type == 'cuda' else None)
            for row, (specfile, spec) in enumerate(cont):
                t_dec = time.monotonic()
                frames = decode_tokens(engine, gen[row])            # one detokenizer call, 12 frames
                sync(engine)
                native2 = np.concatenate([prefix2, frames], axis=0)
                record = dict(common, schema='wm-calib-bench.vavim.output.v1', status='success', mode='continue', run=spec['run'],
                              name=spec['name'], window=window, split=split, purpose=spec['purpose'], sample=spec['sample'],
                              seed=spec['seed'], seed_rule=vc.B['seed_rule'], inputs=spec['inputs'], **base,
                              batch=dict(window=window, size=len(cont), row=row, members=[s['name'] for _, s in cont],
                                         layout=vc.B['batch_layout']),
                              seconds=dict(window_prefix=t_prefix, batch_generation=gen_seconds, per_sample_share=gen_seconds / len(cont),
                                           frame_step_median=float(np.median(steps)), frame_step_max=float(np.max(steps)),
                                           frame_steps=[float(x) for x in steps], decode=time.monotonic() - t_dec),
                              gpu_memory=peak, generated_token_stats=dict(unique_per_frame_median=float(np.median(
                                  [len(np.unique(f)) for f in gen_np[row]])), max_token=int(gen_np[row].max())))
                write_sample(dests[spec['name']], spec, specfile, native2, prefix10,
                             dict(history_tokens=hist_tokens.cpu().numpy(), context_tokens=burnin.cpu().numpy(),
                                  generated_tokens=gen_np[row]), record)
                print(json.dumps(dict(name=spec['name'], status='success')), flush=True)
        for specfile, spec in recon:
            t_ref = time.monotonic()
            real = vc.decode(vc.safe_relative(a.input_root, spec['inputs']['real10.mp4']['path']), vc.MODEL_WH)
            if len(real) != len(vc.ALL_K) or not np.array_equal(real[:len(vc.HISTORY_K)], hist_u8):
                raise ValueError('reference input does not extend the history')
            fut_tokens = encode(engine, real[[k + 20 for k in vc.FUTURE_K]])     # one encoder call, 12 frames
            frames = decode_tokens(engine, fut_tokens)                            # one detokenizer call, 12 frames
            sync(engine)
            native2 = np.concatenate([prefix2, frames], axis=0)
            record = dict(common, schema='wm-calib-bench.vavim.output.v1', status='success', mode='recon', run=spec['run'],
                          name=spec['name'], window=window, split=split, purpose=spec['purpose'], sample=None, seed=None,
                          inputs=spec['inputs'], **base, reference_rule=vc.B['reference_rule'],
                          seconds=dict(window_prefix=t_prefix, reference=time.monotonic() - t_ref),
                          gpu_memory=(dict(peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
                                           peak_reserved_MiB=torch.cuda.max_memory_reserved() / 2**20)
                                      if engine.device.type == 'cuda' else None))
            write_sample(dests[spec['name']], spec, specfile, native2, prefix10,
                         dict(history_tokens=hist_tokens.cpu().numpy(), context_tokens=burnin.cpu().numpy(),
                              reference_tokens=fut_tokens.cpu().numpy()), record)
            print(json.dumps(dict(name=spec['name'], status='success')), flush=True)
    return time.monotonic() - started


def group_specs(paths, bench, input_root, frozen_file):
    groups = OrderedDict()
    for path in paths:
        spec = validate_spec(json.loads(Path(path).read_text()), bench, input_root, frozen_file)
        groups.setdefault((spec['split'], spec['run'], spec['window']), []).append((path, spec))
    for key, items in groups.items():
        if len({s['name'] for _, s in items}) != len(items) or sum(s['mode'] == 'continue' for _, s in items) > 16:
            raise ValueError('duplicate specs or more than 16 samples in one window batch')
        if len({json.dumps(s['inputs']['history10.mp4']) for _, s in items}) != 1:
            raise ValueError('one window, one input build')
    return groups


def require_rulings(groups):
    """Main (DEV/TEST) runs need every pending decision ruled; smoke and diagnostic runs may use the proposed defaults."""
    if any(s['purpose'] == 'main' for items in groups.values() for _, s in items) and vc.PINS.get('decisions_pending'):
        raise ValueError('main runs need every decision ruled (vavim_pins.json decisions_pending is not empty); no GPU started')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--specs', type=Path, nargs='+', required=True)
    ap.add_argument('--bench', type=Path, required=True)
    ap.add_argument('--input-root', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--frozen-file', type=Path)
    ap.add_argument('--equivalence-check', action='store_true')
    ap.add_argument('--equivalence-frames', type=int, default=12)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    paths = [a.bench, a.input_root, a.output_root, a.src, a.weights, *a.specs] + ([a.frozen_file] if a.frozen_file else [])
    if not all(p.is_absolute() for p in paths):
        raise ValueError('explicit absolute paths required for every path argument')
    if a.equivalence_frames not in range(1, len(vc.FUTURE_K) + 1):
        raise ValueError('equivalence frames 1..12')
    groups = group_specs(a.specs, a.bench, a.input_root, a.frozen_file)
    vc.verify_source(a.src)
    main_runs = any(s['purpose'] == 'main' for items in groups.values() for _, s in items)
    pending = vc.PINS.get('decisions_pending') or []
    if a.dry_run:
        print(json.dumps(dict(CPU_spec_and_source_hashes='PASS', specs=len(a.specs), windows=len(groups),
                              decisions_pending=len(pending), main_runs=main_runs, GPU_started=False)))
        return
    require_rulings(groups)
    if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
        raise ValueError('launch through the approved GPU wrapper (CUDA_VISIBLE_DEVICES must be set)')
    if not (a.weights / 'download-verified.json').is_file():
        raise ValueError('owner-approved pinned weight download proof required')
    engine = load_official(a.src, a.weights, 'cuda')
    torch = engine.torch
    common = dict(code_commit=vc.PINS['code']['commit'], weights=dict(release=vc.PINS['weights']['release']['tag'],
                  checkpoint=vc.PINS['weights']['extracted']['member'], checkpoint_sha256=vc.PINS['weights']['extracted']['sha256']),
                  model_load_seconds=engine.record['load_seconds'], gpu=gpu_record(torch), software=software(torch),
                  backends=backend_record(torch), load=engine.record, deviations_from_released_inference=DEVIATIONS,
                  runner_sha256=vc.sha(Path(__file__)), pins_sha256=vc.sha(vc.PINS_PATH))
    print(json.dumps(dict(loaded=True, seconds=round(engine.record['load_seconds'], 1), gpu=common['gpu']['name'])), flush=True)
    first = True
    for (split, run, window), items in groups.items():
        try:
            seconds = run_window(engine, items, a, common, a.equivalence_check and first, a.equivalence_frames)
        except BaseException as error:
            for _, spec in items:
                dest = vc.safe_relative(a.output_root, spec['output_dir'])
                if dest.is_dir() and not (dest / 'sample_outputs.json').exists() and not (dest / 'failure.json').exists():
                    vc.new_json(dest / 'failure.json', dict(error_type=type(error).__name__, error=str(error)[:2000], failed=True,
                                                            no_retry=True, all_partial_outputs_retained=True))
            raise
        first = False
        print(json.dumps(dict(window=window, run=run, specs=len(items), seconds=round(seconds, 1))), flush=True)


if __name__ == '__main__':
    main()
