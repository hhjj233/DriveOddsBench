# {package}: cluster input package of Wan2.1-VACE-1.3B for bench-v1 (generation size {size}, {size_name})

Built by `wm-calib-bench/models/wan21_vace13b/build_input_package.py` on the research server ({created}). Usage:
`wm-calib-bench/cluster/wan21_vace13b/RUNBOOK.md`. There is one package per generation size: `wan21-vace13b-512-inputs-v1` (512×288, the
evaluation canvas) and `wan21-vace13b-832-inputs-v1` (832×480, the official 480P size of the 1.3B model); the submit file selects one with
`SIZE=512` or `SIZE=832`.

The AV2 camera frames have no face or licence-plate processing: keep this package and all outputs in private storage only, never in git, never
public; use is subject to the Argoverse 2 data licence.

## Files

| File | Content |
|---|---|
| `{package}-dev.tar` | {dev_windows} DEV windows: {dev_chunks} chunks (at most {wpc} windows each) + smoke chunk `dev-smoke` ({smoke}: prefix, reference, s0, s1); {dev_specs} specs |
| `{package}-test.tar` | {test_windows} TEST windows: {test_chunks} chunks, **frames up to t0 only**; {test_specs} specs (per window: prefix + 16 continuations) |
| `*.tar.sha256`, `SHA256SUMS` | sha256 of the outer tars; after transfer, run `sha256sum -c SHA256SUMS` first |
| `manifest-{{dev,test}}.json` | per window: file sha256, chunk list |
| `MANIFEST-{{dev,test}}.sha256` (in the outer tar) | sha256 of every chunk tar, of the prompt cache and of every unpacked file (relative to the input root) |
| `prompt-cache/vace-t5-long.safetensors` (+ `.json`, in both outer tars) | output of the official T5 for the fixed prompt and negative prompt (sha256 `{cache_sha}`), shared by all jobs, so jobs need no T5 weights |
| `queues/*.txt` | `dev-smoke`, `dev`, `test` (all), `test-a` ({half_a}), `test-b` ({half_b}) |
| `names-{{dev,test}}-gen.txt` | window lists; their sha256 match Sec. 1 of protocol-v1.md |

Both outer tars unpack into the same directory `{package}/`. Each `chunks/{package}-<chunk>.tar` unpacks into one input root:

- `windows/<split>/<window>/history16_{tag}.mp4`: 33 frames (t0 − 2.0 s … t0, 16 fps), lossless;
- `windows/dev/<window>/real16_{tag}.mp4`: 129 frames (t0 − 2.0 … t0 + 6.0 s), DEV only, for the real reference only;
- `windows/<split>/<window>/frames16_{tag}.json`: the 16 fps frame table (camera frame and timestamp of every frame), check records, file sha256;
- `specs/<split>/<window>/<run>__<name>.json`: one spec per sample (prefix, recon for DEV, s0…s15);
- `chunk-<chunk>.json`: the windows of this chunk, file sha256, spec list and generation size.

## Derivation

- Frames: the measurement band of the source image (AV2 ring_front_center, 1550×2048), rows [797, 1669), full width, resized directly to the
  generation size with cv2.INTER_AREA (at 512×288 this is the bench canvas itself; at 832×480 the image is narrowed horizontally by about 2.5 %,
  and output frames are resized back to 512×288 with INTER_AREA).
- 16 fps time grid: frame j takes the 20 Hz camera frame nearest to t0 + (j − 32)/16 s, and the even offset (the bench's 10 Hz frame) when it falls
  exactly between two frames. Every frame is a real camera frame, off by at most 25 ms and without drift; the 2 s history uses 33 of the 41 camera
  frames (one in every five is skipped). The bench's 10 Hz tick k is frame 32 + round(1.6 k), which is exactly the bench's camera frame c0 + 2k.
- TEST windows render and write only the frames up to t0, with no real future. The TEST real reference is generated on the research server after the
  freeze; the cluster has no `test-recon` queue.
- The seeds of the specs come from `make_bench_nano_specs.seed_of` (one seed table for all models): round 1 uses `seed_of(w, k)`, round 2
  `seed_of(w, 'k|r2')`.

## Notes

- TEST chunks run only with `ALLOW_TEST=1`, after this model's freeze file `freezes/wan21_vace13b-v1.json` is in the repository.
