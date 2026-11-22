# Frozen files

The instrument and the analysis were frozen before any test clip was measured, and each freeze record lists the SHA-256 of its files. The whole-trajectory analysis was frozen later, before any trajectory statistic was computed; its freeze records are in `scripts/trajectory-calibration-v1*/`, and its scripts refuse to run if a listed file changed. The table gives these hashes. Files marked *identical* are byte for byte the frozen files. The three files marked *paths* differ only in the lines that held absolute paths on the machine where they ran: those lines now read the working directory from `$DOB_ROOT` (and `import os` was added where it was missing). No other line differs.

To check a copy against its frozen hash, hash it with `sha256sum`; for the three *paths* files, restore the original path lines first.

| File | Freeze record | Frozen SHA-256 | This copy |
|---|---|---|---|
| `scripts/detect_bench_v2.py` | instrument | `8b8aff62b2bb58aed47ebf5a39b7cf1154f09ac4f13ecd8e090f25c06faf6ca3` | paths (1 line) |
| `scripts/measure_bench_v2.py` | instrument | `064180cdd357b0f3a21e533edb23de9d18a4777f1e0361ea13a271461be3f21d` | identical |
| `scripts/measure_bench_ego_vo.py` | instrument | `745610a31a6866f92e0c08c7d614dde582d3fea1ee928e2844a801a675f0efee` | identical |
| `scripts/measure_ego_motion_raft.py` | instrument | `654ad6d9972c370ee54bc1962ce39f920fb1323a17dcadfea15e7df9739c78ac` | paths (1 line) |
| `instrument_config/config-v2.json` | instrument | `a250a407dbc21881671540588ea7d3f19b4f40a0f16836542e91b81a1c50c2ac` | identical |
| `instrument_config/scorability-rules.json` | instrument | `7a592a66fc719e2b6881df3b823fb40eba2beebc22d01db1fc20e99876a74710` | identical |
| `scripts/analyze_bench_v1.py` | analysis | `a98cdb71c62d5b9228984a01b2e868d8c94427524d94e1de104458fe09bf13aa` | paths (2 lines) |
| `scripts/analyze_wm_calibration.py` | analysis | `a20d55f92f68669b0c55df2896e22814254f856f38bf9853a81fea8560b72ebb` | identical |
| `scripts/check_analyze_bench_v1.py` | analysis | `9131893086f629f51cd413a6567dca5c0debb6aad6e46dff8fe799bc830a204a` | identical |
| `scripts/analyze_bench_v12.py` | analysis v1.2 | `c72a22e3d4c7d4b1bc6c7b57567717cef46b944113445318c6892648148fea1e` | identical |
| `scripts/check_analyze_bench_v12.py` | analysis v1.2 | `9d106163f4fa208bc4d88fbafff4e48416170f89e8f494c4dccd64894d3863c1` | identical |
| `scripts/trajectory-calibration-v1/PLAN-ADDENDUM-1.md` | frozen-addendum-1.txt | `cbb7a139bb801b67b5208c743d3d9f43ff3233be71dfc1cdeb2033c3824dc942` | identical |
| `scripts/trajectory-calibration-v1/analyze_trajectory_climatology.py` | frozen-addendum-1.txt | `bcc15bffdca9a6484a344dd8c126b5dda89e132d17e92b9baecc3c53e18e101a` | identical |
| `scripts/trajectory-calibration-v1/PLAN.md` | frozen.txt | `ac5e868a4c87183c0cc67b3b6aab21a3d839283a5167e8264f3eaf08fe02a557` | identical |
| `scripts/trajectory-calibration-v1/analyze_trajectories.py` | frozen.txt | `edf9d2922af829dde3d96d5ff6ce745e91e32be90ea4e9e016ebd5109149ea0a` | identical |
| `scripts/trajectory-calibration-v1-ext1/PLAN-EXT-1.md` | frozen-ext-1.txt | `af01d421172ab6dcfbe863035fadae68ae1cc0643607b22a5944aa27c87aecba` | identical |
| `scripts/trajectory-calibration-v1-ext1/run_ext1.py` | frozen-ext-1.txt | `d7c6de742f630f5e407d44d8fd995502ef2e49e7989775f0842a73fd21c7bff4` | identical |
| `scripts/trajectory-calibration-v1-ext2/PLAN-EXT-2.md` | frozen-ext-2.txt | `ef17dc72ca2936d3e9c9988ac0f919833c2001edb904ae83040ac52a581f04ea` | identical |
| `scripts/trajectory-calibration-v1-ext2/run_ext2.py` | frozen-ext-2.txt | `3553213d713a2ad6b2345d6fbcd821b326b34b14131b1408fd2046928378f226` | identical |

The per-model freeze records also pin shared scripts that the generation, measurement and analysis of the models ran. The copies of `scripts/bench_holm_two_sided.py`, `scripts/continuation_break.py`, `scripts/derive_bench_quantities.py` and `scripts/validate_bench_v2.py`, and of `ci_core.py`, `derive_canvas.py`, `measure_canvas.py`, `regress_canvas.py`, `rules_canvas.py`, `tables_canvas.py` and `validate_canvas.py` in `scripts/canvas_instrument/`, are identical to the pinned files, as is `scripts/rules_bench_v2.py`, which the instrument protocol lists. `scripts/canvas_instrument/detect_canvas.py`, `scripts/canvas_instrument/ego_vo_band.py`, `scripts/make_bench_nano_specs.py` and `scripts/make_bench_edge_specs.py` differ from the pinned files only in one line that held a machine path (and, in the last two, `import os`).

The Cosmos3 runners `scripts/run_cosmos3_calibration.py` (Cosmos3-Nano, frozen sha256 `c59f30fd39821fd40459c93c239e26ac94be142eae375a5a84cd9a1ad07bdddd`) and `scripts/edge/run_cosmos3_calibration.py` (Cosmos3-Edge, frozen sha256 `ea9bc6d3f03a0392269d03596821de3af48c9352010337ac301fa34c79ae4d84`) differ from their frozen copies only in the checkpoint and source path lines, which read `$DOB_ROOT`. `generation/vista/canvas_spec_vista.json` is identical to the copy pinned by the Vista freeze record (sha256 `161aaef4f769802ea1e5f2a0312f789c85d1d18eae74c8dc78b993e908735ac4`). `scripts/continuation_break_2hz.py` (exploratory, VaViM's continuation breaks on its own 2 Hz grid) reads `scripts/continuation_break.py` next to it.

The `status`, `note` and `decisions_pending` fields of the `*_pins.json` files and the notes of the canvas maps were written during development and are kept as recorded. The settings each model ran with are those of its freeze record; Vista generated TEST at 1024×576 from `vista-inputs-v4-native`.

The model kits in `generation/` follow the per-model freeze records of the protocol (container image, input package and code hashes in each `*_pins.json`). The kits here differ from the frozen kits in comments, help texts and the name of one configuration key (`cluster`): cluster names were removed. A kit records the hashes of its own files when it builds inputs, so these copies are consistent with themselves.
