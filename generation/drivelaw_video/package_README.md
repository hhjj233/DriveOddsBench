# DriveLaW-Video kit

This kit generated the DriveLaW-Video results of the paper; its protocol was frozen on DEV before any TEST clip was generated.
The released source at
`243e0e41148bdb1ae39ce1adf17d026e7cbd4348` omits
`DriveLaW-Video/Infer/diffusers/src/diffusers/models`, although its pipeline imports
the VAE and Transformer there. We reconstruct those 132 files from upstream diffusers
`ebf891a254de807bd88aba8a09792c82df74f459`, which matches all 697 unmodified files that the release ships;
the authors' condition pipeline stays unchanged. The pins and the protocol record the other, equally
matching base and the exact two-file difference. The source hashes and an AST check of eight API calls
pass. The LTX-Video 2B 0.9.5 Open RAIL-M licence was accepted before the download and installation. All 19
weight hashes, the exact environment and the strict transformer and VAE loads passed their checks. Each new
installation needs its own licence approval record (below).

## Files and configuration

`drivelaw_pins.json` pins the upstream inference files and all 19 weight/config
files at Hugging Face `tz2026/DriveLaW`, revision
`376623047ced60a46244241c50b0319713deba29`. The video subset is 29,236,748,364 bytes.
The hash lock resolves the Python 3.10 / torch 2.6.0 cu124 inference
environment. It was installed and validated with the model in the DEV smoke test (59 locked
distributions plus the registered bundled diffusers). No FlashAttention
package or substitute model implementation is added. The smoke records enabled
PyTorch SDPA backends and the model's attention processor classes.

Paths are explicit CLI arguments; `local.env.example` is documentation only. No
script loads a `local.env`, token or credential file. Use a private output root;
data, videos, weights, manifests containing machine paths, and approval records
stay outside git. Each run requires new output/ledger paths and retains failures.

## CPU preparation

Set `BENCH`, `WORK`, `CPU_PY`, `SEED_SNAPSHOT` and `KIT` to private directories.
The CPU interpreter needs AV2, pyarrow, NumPy, OpenCV and ffmpeg. Input preparation
does not import the inference environment.

```bash
"$CPU_PY" -B "$KIT/check_cpu.py"
"$CPU_PY" -B "$KIT/prepare_source.py" --dest "$WORK/source"
"$CPU_PY" -B "$KIT/download_drivelaw_weights.py" --dest "$WORK/weights" --print
# names-dev-smoke.txt contains exactly the two registered DEV smoke window names.
"$CPU_PY" -B "$KIT/build_drivelaw_inputs.py" --bench "$BENCH" --split dev \
  --names "$WORK/names-dev-smoke.txt" --out "$WORK/inputs"
"$CPU_PY" -B "$KIT/make_drivelaw_specs.py" --bench "$BENCH" \
  --input-root "$WORK/inputs" --split dev --names "$WORK/names-dev-smoke.txt" \
  --mode continue --run drivelaw-dev-smoke-v1 --samples 0-1 \
  --snapshot "$SEED_SNAPSHOT" --out "$WORK/specs"
"$CPU_PY" -B "$KIT/make_drivelaw_specs.py" --bench "$BENCH" \
  --input-root "$WORK/inputs" --split dev --names "$WORK/names-dev-smoke.txt" \
  --mode recon --run drivelaw-dev-smoke-ref-v1 \
  --snapshot "$SEED_SNAPSHOT" --out "$WORK/specs"
```

For normal generation `--samples 0-15` is the shared K=16 seed table. The smoke
uses 2 DEV windows x 2 seeds, plus 2 real references. This preparation reads no TEST
window. Generation and measurement have separate TEST
freeze guards.

## Approved source, licence and environment

`prepare_source.py` reconstructs models/ only from the selected base.
The source guard rejects unregistered or changed Python files. Preserve the
shipped condition pipeline. The environment/GPU paths validate the complete
source; strict checkpoint loading must still prove compatibility after download.

The licence approval record is private and is written only after the licence is explicitly accepted. The downloader
requires these fields:
`owner_approved: true`, `approved_by: "owner"`, the exact `repo`, `revision`, and
`license_sha256` from the pins. It checks approval before mkdir/network, verifies
all file bytes/hashes, and refuses to replace mismatched or partial files.

With the approval record in place, download the weights and build the environment:

```bash
"$CPU_PY" -B "$KIT/download_drivelaw_weights.py" --dest "$WORK/weights" \
  --license-approval "$WORK/owner-license-approval.json"
bash "$KIT/prepare_env.sh" "$UV" "$PYTHON310" "$WORK/source" "$WORK/venv"
```

## Local smoke

`run_local_smoke.py --print` audits a sequential plan without starting the GPU.
Pass all four continuation specs and the two reference specs. It accepts DEV only.
Use the host Python with working `os.pidfd_open` and `signal.pidfd_send_signal` for
supervision, and the new environment's Python for science. The wrapper path and
SHA256 must name the unchanged GPU wrapper, whose lock and free-memory
policy apply.

```bash
"$HOST_PY" -B "$KIT/run_local_smoke.py" --science-python "$WORK/venv/bin/python" \
  --bench "$BENCH" --input-root "$WORK/inputs" --output-root "$WORK/outputs" \
  --src "$WORK/source" --weights "$WORK/weights" \
  --wrapper "$GPU_WRAPPER" --wrapper-sha256 "$GPU_WRAPPER_SHA256" \
  --ledger "$WORK/ledger-smoke-v1" --specs "${SPECS[@]}" --print
```

To start the jobs, change `--print` to `--run`. Every spec is one job:
30-minute science timeout, 30-minute wrapper memory wait, 61-minute outer bound;
no automatic retries. The default VRAM request is 20,000 MiB (the measured peak reserved, 17,546 MiB, plus
2,000, rounded up to 1,000), plus the wrapper's 6,144 MiB margin, with host MemAvailable >= 36 GiB.
The supervisor binds its own child
PID/birth/session, retains logs and failures, and reaps only its own descendants.

Run `check_smoke.py` on the six returned sample directories. Its report covers the measured wall
time, peak allocated and reserved VRAM, format, provenance, seed diversity and seam
diagnostics. The full DEV check (A1) before the stage-1 freeze also reports the VO time scale over 0-0.4 and
0.4-0.8 s (registered guard: within [0.8, 1.25]), repeated-step effects, the t0 jump J, the <=5%
window/family drop gate, the window-rule export and the clip-rule sensitivity. The smoke test below is not that check.

## Registered model behaviour

Raw camera nearest timestamps -> crop rows [797,1669) -> native 1280x704 @8 Hz.
The model receives 9 real frames over the last 1 s. Caption numbers use only
2 s of real past poses; labels remain literal training 2 s/4 s; no route/turn
command. Planar speed and acceleration are disclosed finite-difference proxies.
Two unchanged official 33-frame calls use a verbatim prompt; pass 2 takes the
last 9 generated frames, with the pass-1 seed + 1000003. The official last-frame anchor and
noise reinjection (beta 1, sigma 0.3, first 3 steps) are kept. No frame_rate is passed, so the official temporal
default of 25 applies; output timing remains native 8 Hz.

Native65 = fixed seeded VAE prefix[:8] + pass1[0:33] + pass2[9:33]. Reference
segments are real frames [-1,3] and [2,6] seconds through the same sampled VAE,
normalisation/decode_timestep0/tiling and assembly. The decoder is noncausal;
decoded history may depend on later frames in its segment. Use per-clip t0.
Nearest-native 8->10 mapping yields the 81-frame 512x288 canvas, <=50 ms error,
exact 1/2/4/6 s horizons, with 12 repeated steps among 60 future canvas steps.
The new canvas spec is supplied without modifying any instrument code.

## Smoke test on two DEV windows

Two DEV windows x 2 seeds plus 2 references: all 6 science jobs exited with code 0 and left their own
sessions empty. Structural checks passed 6/6: 81 canvas frames at 10 Hz, 65 native frames at 8 Hz, two 33-frame
passes, hashes, settings, strict checkpoint loads and exact assembly and mapping. Every 48-frame native future had
48 unique frames, and the two seeds differed after t0. Peak GPU memory on the RTX 6000 Ada: 17,546 MiB reserved,
16,437 MiB allocated. Continuation processing took 53.52-62.83 s (excluding model start), single diffusion calls
22.82-27.63 s and complete wrapper jobs 48.92-93.57 s. Attention: the native SDPA dispatcher with
LTXVideoAttnProcessor, with the Torch flash, efficient and math backends enabled; the kernel selected per call was
not profiled.

On inspection, the four DEV futures show no obvious scene cut at t0 or 3 s, but low motion and less scene and
traffic evolution than their real references (mean RGB step between native frames 0.258-1.072, against 4.828 and
9.414 for the references). Continuation breaks (descriptive): median J 1.36 (0 of 4 above 3), median C 0.58
(0 of 4 above 2).
