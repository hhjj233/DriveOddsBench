#!/usr/bin/env bash
# Download the pinned Wan-AI/Wan2.1-VACE-1.3B checkpoint into a local directory (research server; the cluster staging is
# cluster/wan21_vace13b/stage_weights.sh). Resumable: a finished file whose size and sha256 match is kept; a partial file is
# continued. Every file is checked against the size and sha256 below (Hugging Face LFS sha256 of the pinned revision for
# the large files; for the small files the sha256 of the download whose git blob sha1 matched the Hub tree, 2026-11-05).
# Wan-AI/Wan2.1-VACE-1.3B is Apache-2.0 and not gated: no token, no licence step. Needs bash, curl, sha256sum.
#   bash download_vace_weights.sh DEST [--no-t5]
# --no-t5 leaves out the 11.4 GB umT5-XXL encoder (only needed to make the prompt cache, make_vace_prompt_cache.py).
# Not downloaded: assets/ (model-card pictures) and .gitattributes.
set -euo pipefail
DEST="${1:?usage: download_vace_weights.sh DEST [--no-t5]}"
WITH_T5=1
[ "${2:-}" = --no-t5 ] && WITH_T5=0
REV=574e6a744642ce3bee319afc31496b88bde8aac4   # Wan-AI/Wan2.1-VACE-1.3B revision (models/wan21_vace13b/vace_pins.json)
BASE="https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B/resolve/$REV"
mkdir -p "$DEST/google/umt5-xxl"
cd "$DEST"

fetch() {  # <path in the HF repo = path in DEST> <bytes> <sha256>
  local rel="$1" bytes="$2" sha="$3"
  if [ -f "$rel" ] && [ "$(stat -L -c %s "$rel")" = "$bytes" ] && echo "$sha  $rel" | sha256sum -c --status; then
    echo "present and verified: $rel"; return
  fi
  curl -fL --retry 8 --retry-delay 20 -C - -o "$rel.part" "$BASE/$rel"
  [ "$(stat -c %s "$rel.part")" = "$bytes" ] || { echo "size mismatch for $rel" >&2; exit 1; }
  echo "$sha  $rel.part" | sha256sum -c - || { echo "sha256 mismatch for $rel" >&2; exit 1; }
  mv "$rel.part" "$rel"
}

fetch diffusion_pytorch_model.safetensors 7146067848 c46a6f5f7d32c453c3983bbc59761ea41cd02ad584fb55d1a7ee2b76145847a2
fetch Wan2.1_VAE.pth 507609880 38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981
fetch config.json 349 e19df24acf3f440e0aea37d299f22b96f88d164f3349ae696daecd960a35ab1e
fetch LICENSE.txt 11357 c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4
fetch README.md 40485 1eed52faea991fffe9c4bbd3f2df4e7ba376a05b1ab87ce734b0d2af18fd4aff
fetch google/umt5-xxl/special_tokens_map.json 6623 7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d
fetch google/umt5-xxl/spiece.model 4548313 e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458
fetch google/umt5-xxl/tokenizer.json 16837417 6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b
fetch google/umt5-xxl/tokenizer_config.json 61728 ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05
if [ "$WITH_T5" = 1 ]; then
  fetch models_t5_umt5-xxl-enc-bf16.pth 11361920418 7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d
fi

# SHA256SUMS of what is present (the adapter compares it with the pins)
find . -type f ! -name SHA256SUMS ! -name REVISION ! -name '*.part' ! -path './.cache/*' -printf '%P\n' | LC_ALL=C sort | xargs sha256sum > SHA256SUMS
echo "Wan-AI/Wan2.1-VACE-1.3B@$REV (Apache-2.0, not gated)" > REVISION
cat SHA256SUMS
echo "done: $(du -sh "$DEST" | cut -f1) in $DEST"
