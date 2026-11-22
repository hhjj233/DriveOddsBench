# orbis2-inputs-v1: input package of Orbis 2 for bench-v1

Built from the bench-v1 windows by `wm-calib-bench/models/orbis2/build_input_package.py` on the research server ({created}). Usage:
`wm-calib-bench/cluster/RUNBOOK.md`.

## Files

| File | Content |
|---|---|
| `orbis2-inputs-v1-dev.tar` | {dev_windows} DEV windows: {dev_chunks} chunks (at most {wpc} windows each) + 1 smoke chunk `dev-smoke` ({smoke_count} windows: {smoke}) |
| `orbis2-inputs-v1-test.tar` | {test_windows} TEST windows: {test_chunks} chunks (at most {wpc} windows each) |
| `*.tar.sha256`, `SHA256SUMS` | sha256 of the outer tars; after transfer, check them first with `sha256sum -c SHA256SUMS` |
| `manifest-{{dev,test}}.json` | per window: source, derivation and the sha256 of every file, plus the chunk list |
| `MANIFEST-{{dev,test}}.sha256` | sha256 of every file of an unpacked chunk (relative to the input root) and of every chunk tar |
| `queues/*.txt` | chunk lists for the batch system's `queue chunk from ...`: `dev-smoke`, `dev`, `test-a`, `test-b` |
| `names-{{dev,test}}-gen.txt` | window lists; their sha256 match Sec. 1 of protocol-v1.md |

Both outer tars unpack into the same directory `orbis2-inputs-v1/`. Each `chunks/orbis2-inputs-v1-<chunk>.tar` unpacks into one input root:

- `windows/<split>/<window>/history.mp4`: 21 frames (t0 − 2.0 s … t0), 512×288, 10 fps;
- `windows/<split>/<window>/real.mp4`: 81 frames (t0 − 2.0 … t0 + 6.0 s), 512×288, 10 fps;
- `specs/<split>/<window>/<window>-s{{0..15}}.json`: the 16 continuation samples;
- `specs/<split>/<window>/<window>-recon.json`: the real reference;
- `chunk-<chunk>.json`: the windows of this chunk and the sha256 of its files.

## Derivation

- `real.mp4` is a byte-for-byte copy of bench-v1 `windows-<split>/<window>/canvas/real.mp4` (sha256 checked).
  It is the measurement band (source rows [797, 1669), full width) resized to 512×288 with INTER_AREA, i.e. the evaluation canvas.
- `history.mp4` is frames 0–20 of the same file, decoded to rgb24 and re-encoded with the rendering encoder (libx264rgb -crf 0 -preset medium);
  the decoded frames were checked to equal the source frames. Only these 21 frames are decoded at build time.
- The seeds of the specs come from `make_bench_nano_specs.seed_of` (one seed table, shared with Cosmos3-Nano). Continuation specs reference only
  `history.mp4`; only the recon spec references `real.mp4`.

## Notes

- TEST windows include the real future, used only for the real reference (tokenizer round trip). No TEST clip is viewed, and TEST clips
  are measured only with the frozen instrument.
- The frames are raw Argoverse 2 camera images without face or licence-plate processing: keep them in private storage only,
never in git, never public. Use is subject to the Argoverse 2 data licence (non-commercial, attribution).
