# epona-inputs-v1: cluster input package of Epona for bench-v1

Built by `wm-calib-bench/models/epona/build_input_package.py` on the research server ({created}). Usage:
`wm-calib-bench/cluster/epona/RUNBOOK.md`.

## Files

| File | Content |
|---|---|
| `epona-inputs-v1-dev.tar` | {dev_windows} DEV windows: {dev_chunks} chunks (at most {wpc} windows each) + smoke chunk `dev-smoke` ({smoke}) |
| `epona-inputs-v1-test.tar` | {test_windows} TEST windows: {test_chunks} chunks (at most {wpc} windows each), **history only** |
| `*.tar.sha256`, `SHA256SUMS` | sha256 of the outer tars; after transfer, run `sha256sum -c SHA256SUMS` first |
| `manifest-{{dev,test}}.json` | per window: file sha256, chunk list, spec list (hash of the seed table) |
| `MANIFEST-{{dev,test}}.sha256` | sha256 of every file of an unpacked chunk (relative to the input root) and of every chunk tar |
| `queues/*.txt` | chunk lists for `queue chunk from ...`: `dev-smoke`, `dev`, `test` (all), `test-a` ({half_a}), `test-b` ({half_b}) |
| `names-{{dev,test}}-gen.txt` | window lists; their sha256 match Sec. 1 of protocol-v1.md |

Both outer tars unpack into the same directory `epona-inputs-v1/`. Each `chunks/epona-inputs-v1-<chunk>.tar` unpacks into one input root:

- `windows/<split>/<window>/history.mp4`: 11 frames (t0 − 2.0 s … t0, 5 Hz), 1024×512, lossless;
- `windows/<split>/<window>/past_poses.json`: ego poses at the 11 ticks up to t0 (AV2 city_SE3_egovehicle, in the ego frame at t0);
- `windows/dev/<window>/real.mp4`: 41 frames (t0 − 2.0 … t0 + 6.0 s), DEV only, for the real reference only (DCAE round trip);
- `windows/<split>/<window>/build.json`: rendering and check records;
- `specs/<split>/<window>/<window>-s{{0..15}}.json`: the 16 continuation samples, bound to batch layout 16 (one window per batch);
- `specs/dev/<window>/<window>-recon.json`: the DEV real reference;
- `chunk-<chunk>.json`: the windows of this chunk, file sha256, batch size and image band.

## Derivation

- Image band: rows [579, 1451) of the source image (AV2 ring_front_center, 1550×2048), full width, resized to Epona's 1024×512 with cv2.INTER_AREA
  (chosen so that the horizon is centred). The same code, applied to the measurement band [797, 1669) and resized to 512×288, reproduces
  bench-v1 `canvas/real.mp4` frame for frame (checked; for TEST only on the frames up to t0).
- 5 Hz time grid: the bench's 10 Hz ticks k = −20, −18, … (each tick takes the nearest 20 Hz camera frame, as in window.json).
- TEST windows are built with `--history-only`: only frames and poses with k ≤ 0 are rendered and written, with no real.mp4 and no frame or tick
  after t0. The TEST real reference is generated on the research server after the freeze.
- The seeds of the specs come from `make_bench_nano_specs.seed_of` (one seed table for all models); continuation specs reference only
  `history.mp4` and `past_poses.json`.

## Notes

- The frames are raw Argoverse 2 camera images without face or licence-plate processing: keep them in private storage only,
never in git, never public. Use is subject to the Argoverse 2 data licence (non-commercial, attribution).
- TEST chunks are run only after the Epona protocol is frozen (`bench-v1/protocol-epona-frozen.txt` exists).
