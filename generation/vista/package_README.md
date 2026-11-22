# {package}: cluster input package of Vista for bench-v1 (generation size {size})

Built by `wm-calib-bench/models/vista/build_input_package.py` on the research server ({created}). Usage:
`wm-calib-bench/cluster/vista/RUNBOOK.md`. Two generation sizes were prepared: `vista-inputs-v3` (320×576: height and width are multiples of 64, the UNet code is
unchanged, and output frames are resized to the 512×288 evaluation canvas with INTER_AREA) and `vista-inputs-v4-native` (Vista's
training size 576×1024, output frames resized to the canvas by an exact 2×2 area average). The frozen protocol generated TEST with
`vista-inputs-v4-native`. Earlier packages, kept unchanged: `vista-inputs-v2` (288×512) and `vista-inputs-v1`
(576×1024, old specs). The generation size of this package is in the title.

## Files

| File | Content |
|---|---|
| `{package}-dev.tar` | {dev_windows} DEV windows: {dev_chunks} chunks (at most {wpc} windows each) + smoke chunk `dev-smoke` ({smoke}) |
| `{package}-test.tar` | {test_windows} TEST windows: {test_chunks} chunks (at most {wpc} windows each), **frames up to t0 only** |
| `*.tar.sha256`, `SHA256SUMS` | sha256 of the outer tars; after transfer, run `sha256sum -c SHA256SUMS` first |
| `manifest-{{dev,test}}.json` | per window: file sha256, chunk list, spec list (hash of the seed table) |
| `MANIFEST-{{dev,test}}.sha256` (in the outer tar) | sha256 of every chunk tar and of every unpacked file (relative to the input root) |
| `queues/*.txt` | chunk lists for `queue chunk from ...`: `dev-smoke`, `dev`, `test` (all), `test-a` ({half_a}), `test-b` ({half_b}) |
| `names-{{dev,test}}-gen.txt` | window lists; their sha256 match Sec. 1 of protocol-v1.md |

Both outer tars unpack into the same directory `{package}/`. Each `chunks/{package}-<chunk>.tar` unpacks into one input root:

- `windows/<split>/<window>/history21_<W>.mp4`: 21 frames (t0 − 2.0 s … t0, 10 Hz) at the generation size (<W> is the generation width: 576 for
  320×576, 1024 for 576×1024), lossless. Frame 20 is the anchor frame (the only frame the model sees); frames 0–19 serve only the prefix
  (the real history after a round trip through Vista's VAE, used as frames 0–19 of every video);
- `windows/dev/<window>/real81_<W>.mp4`: 81 frames (t0 − 2.0 … t0 + 6.0 s) at the generation size, DEV only, for the real reference only;
- `windows/dev/<window>/actions_t2.json`: trajectory actions for track T2 (DEV only; the default tracks of this package are {tracks}, T2 is off by default);
- `windows/<split>/<window>/inputs.json`: rendering and check records;
- `specs/<split>/<window>/*.json`: one spec per sample (prefix, recon for DEV, s0…s15 for h0), memory layout `official`, generation size as in the title;
- `chunk-<chunk>.json`: the windows of this chunk, file sha256, spec list, memory layout and generation size.

## Derivation

- Frames: the measurement band of the source image (AV2 ring_front_center, 1550×2048), rows [797, 1669), full width, resized directly to the
  generation size with cv2.INTER_AREA. 320×576: 576×320 has aspect ratio 1.8 against 1.7775 for the band, a vertical squeeze of about 1.3 %
  (a disclosed deviation); the adapter resizes every output frame to 512×288 with INTER_AREA, bit-identical to cv2.INTER_AREA and checked on every
  input frame. 576×1024: 1024×576 is exactly twice the canvas; output frames are resized to the canvas by an exact 2×2 area average (equal to
  cv2.INTER_AREA) and then differ from the bench's `canvas/real.mp4` by at most one grey level.
- 10 Hz time grid: bench ticks k = −20…60 (each tick takes the nearest 20 Hz camera frame, as in window.json).
- TEST windows render and write only frames with k ≤ 0: no real future and no future pose. The TEST real reference is made after the freeze, in the
  add-on package `{package}-test-recon.tar` (built by `--recon-addon test` of the same script), which holds the real frames rendered after the freeze
  and the prefixes already generated on the cluster.
- The seeds of the specs come from `make_bench_nano_specs.seed_of` (one seed table for all models and versions); continuation specs reference only
  `history21_<W>.mp4`.

## Notes

- The frames are raw Argoverse 2 camera images without face or licence-plate processing: keep them in private storage only,
never in git, never public. Use is subject to the Argoverse 2 data licence (non-commercial, attribution).
- TEST chunks are run only after the Vista protocol is frozen (`bench-v1/protocol-vista-frozen.txt` exists).
