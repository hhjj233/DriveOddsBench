# ltx25d-diag-v1: DEV diagnostic package of the distilled LTX-2.5 (queue dev-diag)

Built by `wm-calib-bench/models/ltx25/build_diag_package.py` on the research server ({created}) from the completed DEV runs
`ltx25d-dev-recon` and `ltx25d-dev-k16`. Usage: step 6.3 of `wm-calib-bench/cluster/ltx25/RUNBOOK.md`.

This job generates no new clip. It decodes the final latents saved with the DEV samples (`latents.safetensors`) again, with the same decoder call
as at generation time:

| Run | Content |
|---|---|
| `ltx25d-dev-recon-draws` | three more DiffVAE decodes of each reference (new seeds `seed_of(w, 'recon-d<j>')`): the noise floor of the decoder |
| `ltx25d-dev-redecode` | each reference and s0 of the first window decoded again from the saved decoder state: must be bit-identical to the DEV clips |
| `ltx25d-dev-conv-recon`, `ltx25d-dev-conv-k16` | the conv decoder on the references and the 16 continuations (each from its saved decoder state): a sensitivity arm |
| `ltx25d-dev-conv-probe` | the reference of the first window through the conv decoder again with another seed: checks whether that decoder is deterministic |

- Windows (10, chosen by a fixed hash rule): {windows}
- Specs: {specs}; chunk tar: `{tar}`
- GPU of the DEV runs: {gpus}. This job must use the same GPU type for the re-decode check to be bit-identical.

## Files

| File | Content |
|---|---|
| `{tar}` | chunk: `latents/<run>/<name>.safetensors`, `specs/dev/<w>/<name>.json`, `chunk-dev-diag.json` |
| `queues/dev-diag.txt` | queue list for the batch system |
| `manifest.json` | window-selection rule, hashes of the source samples, seed table |
| `SHA256SUMS` | sha256 of the files above; after transfer, run `sha256sum -c SHA256SUMS` first |

DEV latents and specs only, with no TEST content. The latents decode into AV2 frames: keep them in private storage only, never in git, never public.
