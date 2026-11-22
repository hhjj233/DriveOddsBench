# VaViM-L kit (2.0 s of 2 Hz context, 12 generated frames at 2 Hz)

This kit generated the VaViM results of the paper; its protocol was frozen on DEV before any TEST clip was generated. It follows the
DrivingWorld / ReWorld-Video kit layout. The model is the released VaViM-L of valeo.ai ("VaViM and VaVAM: Autonomous
Driving through Video Generative Modeling", arXiv 2502.15672): a 1.24 B-parameter GPT-2 (24 layers, width 2048, 16 heads
of 128, muP readout) over the LlamaGen VQ-16 image tokenizer (codebook 16,384, 8-d l2-normalised codes). A 512x288 frame
is 18x32 = 576 tokens; the model was trained on 8-frame clips at 2 Hz (context up to 4,608 tokens). Code
valeoai/VideoActionModel `738050ec7000fc23b7af0c3ca14693d75abb056e` (MIT); weights: GitHub release v1.0.0,
`width_2048_pretrained_139k_total_155k` (VaViM-L, fine-tuned) and the TorchScript tokenizer/detokenizer, research-only
VideoActionModel RAIL-M licence, accepted before the download. No HF token is needed.

## Files

| file | role |
|---|---|
| `vavim_pins.json` | code commit and the git blob of every tracked file; release assets (sizes, sha256), the extracted checkpoint, allowed pickle globals; all bench settings; the decision record |
| `vavim_common.py` | hashes, guards, the 2 Hz grid and its map onto the 10 Hz canvas, the released pixel conventions, shared seeds, lossless video I/O (no torch) |
| `build_vavim_inputs.py` | raw AV2 -> 512x288 canvas renders (= model frames): history (21), context (5), DEV real future (81) |
| `make_vavim_specs.py` | continuation specs (shared seed table, `--samples 0-15`), reference specs; purposes main / smoke / diagnostic |
| `run_vavim_bench.py` | loads the model once, runs every spec (one batch per window), writes `<run>/native/<name>/`; `--equivalence-check` |
| `canvas_spec_vavim.json` | canvas-instrument spec (new file; instrument unchanged) |
| `download_vavim_weights.py`, `prepare_source.py`, `prepare_env.sh`, `requirements-vavim.{in,lock.txt}` | pinned weights, exact source, hash-locked environment (CPython 3.10.19 inside the environment directory) |
| `check_cpu.py`, `check_strict_load.py`, `check_smoke.py` | CPU contract tests; CPU strict load, tokenizer round trip and released/kit equivalence with the real weights; DEV smoke structural checks |

Machine paths go in the git-ignored `local.env` (template `local.env.example`); every tool takes explicit arguments.

## Model behaviour (registered settings)

- Frames: VaViM's 512x288 frame is the bench canvas itself (band rows [797, 1669), full width, cv2.INTER_AREA), so the
  model frame equals canvas/real.mp4 bit for bit (checked by the input builder). No crop, pad, black band or second resize.
- Time: 2 Hz. Context = the history frames at ticks -20, -15, -10, -5, 0 (the whole 2.0 s); 12 generated frames = ticks
  5, 10, ..., 60, by the released `MupGPT2.forward_inference` (KV cache; when 7 frames are in the context the oldest is
  dropped: the #57 fix, so the model always sees at most 3.5 s).
- Sampler: the released `_sample_next_token` arithmetic (temperature, top-k with `sorted=False`, renormalise, multinomial)
  at temperature 1.0 and the registered top-k (the released default is top-1 = greedy, which would make all K samples
  identical); one `torch.Generator` per sample, seeded seed_of(window, k); bf16 autocast for the GPT, the encoder and the
  decoder as in the released evaluation scripts.
- Batch: one window per batch (K = 16 rows, ascending k). At batch 1 the kit path equals the released `forward_inference`
  token for token (CPU test with the real weights; `--equivalence-check` repeats it on the GPU, 12 frames).
- Canvas: vision.mp4 frame i = tick k = i - 20 shows the 2 Hz model frame of tick 5 round(k / 5) (map `nearest`). Shared
  prefix: the VQ round trip of the history (one encoder and one decoder call per window), shown at 2 Hz by the same map.
  Reference: the real future frames at ticks 5..60 through the same codec, same prefix, same map.
- Outputs per sample: vision.mp4 (81 x 512x288 at 10 fps, lossless), native2.mp4 (17 x 512x288 at 2 fps: 5 prefix + 12
  generated frames), tokens.npz (history, context and generated or reference token grids), sample_outputs.json.

## CPU preparation

```bash
source local.env   # or set the variables by hand
bash prepare_env.sh "$UV" "$VAVIM_ENV"                                  # CPython 3.10.19 + hash-locked venv
"$VAVIM_PY" -B prepare_source.py --dest "$VAVIM_SRC"                    # fresh machines only
"$VAVIM_PY" -B download_vavim_weights.py --approval "$VAVIM_WORK/owner-license-approval.json" --dest "$VAVIM_WEIGHTS"
CUDA_VISIBLE_DEVICES= "$VAVIM_PY" -B check_cpu.py --src "$VAVIM_SRC" --snapshot "$WCB_SNAPSHOT_DIR"
CUDA_VISIBLE_DEVICES= "$CPU_PY" -B check_cpu.py --bench "$WCB_BENCH"
"$CPU_PY" -B build_vavim_inputs.py --bench "$WCB_BENCH" --split dev --names "$VAVIM_WORK/names-dev-smoke.txt" \
  --out "$VAVIM_WORK/inputs-dev-smoke-v3"
CUDA_VISIBLE_DEVICES= "$VAVIM_PY" -B check_strict_load.py --src "$VAVIM_SRC" --weights "$VAVIM_WEIGHTS" \
  --input-root "$VAVIM_WORK/inputs-dev-smoke-v3" --windows 2eeaccb0-t2.5 47a9fb77-t9.5 --out NEW.json [--equivalence-frames 1]
for mode in continue recon; do
  "$CPU_PY" -B make_vavim_specs.py --bench "$WCB_BENCH" --input-root "$VAVIM_WORK/inputs-dev-smoke-v3" --split dev \
    --names "$VAVIM_WORK/names-dev-smoke.txt" --mode $mode --run vavim-dev-smoke-v1 --samples 0-1 --purpose smoke \
    --snapshot "$WCB_SNAPSHOT_DIR" --out "$VAVIM_WORK/specs-dev-smoke-v2"
done
```

## GPU smoke test

One process, both windows (batch 2 per window, then the window's reference), through a GPU wrapper:

```bash
"$VAVIM_PY" -B run_vavim_bench.py --specs "$VAVIM_WORK"/specs-dev-smoke-v2/vavim-dev-smoke-v1/*.json --bench "$WCB_BENCH" \
  --input-root "$VAVIM_WORK/inputs-dev-smoke-v3" --output-root "$OUT" --src "$VAVIM_SRC" --weights "$VAVIM_WEIGHTS" \
  --equivalence-check            # add --dry-run first: CPU spec/source validation only
"$CPU_PY" -B check_smoke.py --samples "$OUT"/vavim-dev-smoke-v1/native/* --report NEW.json \
  --equivalence "$OUT"/vavim-dev-smoke-v1/equivalence-check-*.json
```

For K = 16, `--samples 0-15`; the runner batches a window's 16 samples. Main (non-smoke, non-diagnostic) runs are refused
until `decisions_pending` in the pins is empty. Scoped DEV probes on windows outside the 60 generation windows need
`--diagnostic` in the input builder and `--purpose diagnostic` in the spec builder (names-dev.txt; run name
`vavim-dev-diag-*`; refused for TEST). TEST continuation specs need `protocol-vavim-frozen.txt`; TEST reference specs and
TEST future inputs need `protocol-vavim-stage2-frozen.txt`. Raw AV2 frames stay out of git and out of public storage.
