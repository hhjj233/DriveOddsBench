# ltx25d-inputs-v1: input package of the distilled LTX-2.5 for bench-v1

Built by `wm-calib-bench/models/ltx25/build_input_package.py` on the research server ({created}) from the checked LTX input root
`outputs/wm-calibration/ltx25-inputs-v1`. Usage: `wm-calib-bench/cluster/ltx25/RUNBOOK.md`.

## Files

| File | Content |
|---|---|
| `ltx25d-inputs-v1-dev.tar` | {dev_windows} DEV windows: {dev_chunks} chunks (at most {wpc} windows each), plus smoke chunk `dev-smoke` ({smoke_count} windows: {smoke}) |
| `ltx25d-inputs-v1-test.tar` | **histories** of {test_windows} TEST windows: {test_chunks} chunks (at most {wpc} windows each), queues {test_queues} |
| `*.tar.sha256`, `SHA256SUMS` | sha256 of the outer tars; after transfer, run `sha256sum -c SHA256SUMS` first |
| `manifest-{{dev,test}}.json` | per window: file sha256, time offsets, chunk list, spec list (with the hash of the seed table) |
| `MANIFEST-{{dev,test}}.sha256` | sha256 of every file of an unpacked chunk (relative to the input root) and of every chunk tar |
| `queues/*.txt` | chunk lists for the batch system's `queue chunk from ...` |
| `names-{{dev,test}}-gen.txt` | window lists; their sha256 match Sec. 1 of protocol-v1.md |

Both outer tars unpack into the same directory `ltx25d-inputs-v1/`. Each `chunks/ltx25d-inputs-v1-<chunk>.tar` unpacks into one input root:

- `windows/<split>/<window>/history20.mp4`: 41 frames (t0 − 2.0 s … t0), 512×288, 20 fps, the condition of the first stage;
- `windows/<split>/<window>/history20_1024.mp4`: the same 41 frames at 1024×576, the condition of the second stage;
- `windows/dev/<window>/real20_1024.mp4` (DEV only): 161 frames (t0 − 2.0 … t0 + 6.0 s), 1024×576, the input of the real reference;
- `windows/<split>/<window>/frames20.json`: the camera frame and time offset of every frame, and the build-time checks;
- `specs/<split>/<window>/<window>-s{{0..15}}.json`: the 16 continuations; DEV also has `<window>-recon.json` (the real reference);
- `chunk-<chunk>.json`: the windows of this chunk and the sha256 of every file.

## Derivation

- Frame j (j = 0…160) is the 20 Hz frame of the AV2 front camera nearest to t0 + (j − 40) × 0.05 s; even frames are the bench's 10 Hz frames,
  and frame 40 is t0.
- Both sizes are rendered from the source image by the bench's frozen rendering: source rows [797, 1669), full width,
  `cv2.resize(..., INTER_AREA)`. The 512×288 history matches the bench's `canvas/real.mp4` pixel for pixel (checked at build time).
- The seeds of the specs come from `make_bench_nano_specs.seed_of` (one seed table, shared with Cosmos3-Nano and Orbis 2); precision is bf16
  (`generation.quantization = none`). Continuation specs reference only the history; the DEV reference specs reference `real20_1024.mp4`.

## Notes

- **TEST has histories only**: this package holds no TEST frame after t0. The TEST real reference is built after the protocol freeze, as a separate
  add-on package.
- No TEST clip is viewed, and TEST clips are measured only with the frozen instrument.
- The frames are raw Argoverse 2 camera images without face or licence-plate processing: keep them in private storage only,
never in git, never public. Use is subject to the Argoverse 2 data licence (non-commercial, attribution).
