#!/usr/bin/env bash
# Download the pinned LTX-2.5 files of one variant (ltx25_pins.json: weights.files whose "variants" include it and that are
# required; --with-optional adds the conv decoder) at the pinned revision into the weights directory, then check every
# file's size and sha256 against the pins and write SHA256SUMS and weights-manifest.json there (all files present).
#   distilled (default): distilled transformer, x2 spatial upsampler, text encoder, video VAE (DiffVAE), audio VAE (~71 GB)
#   dev:                 dev transformer, text encoder, video VAE, audio VAE (~70 GB)
#   all:                 both (~113 GB)
#
# Needs the Hugging Face terms of Lightricks/LTX-2.5 accepted by the account whose token this server uses (the repo is
# gated "auto"). The script never accepts terms and never prints the token: it stops with the URL if access is missing.
#
#   download_ltx25_weights.sh [--variant distilled|dev|all] [--with-optional] [WEIGHTS_DIR]
# Machine-specific settings (models/ltx25/local.env, template local.env.example): LTX25_ENV (the tools' environment),
# LTX25_MODELS_DIR (default WEIGHTS_DIR) and HF_XET_CACHE (the download's chunk cache, kept off a small home disk).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/local_env.sh"
load_local_env
need_setting LTX25_ENV HF_XET_CACHE
ENV="$LTX25_ENV"
PY="$ENV/venv/bin/python"
VARIANT=distilled
OPTIONAL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    --with-optional) OPTIONAL=1; shift ;;
    *) break ;;
  esac
done
DEST="${1:-${LTX25_MODELS_DIR:-}}"
[ -n "$DEST" ] || { echo "no weights directory: pass WEIGHTS_DIR or set LTX25_MODELS_DIR (models/ltx25/local.env)" >&2; exit 2; }
mkdir -p "$DEST"
# the xet chunk cache stays where local.env puts it (off the home disk); the token stays where `hf auth login` put it
export HF_XET_CACHE
export HF_HUB_DISABLE_TELEMETRY=1

mapfile -t FILES < <("$PY" -I - "$HERE/ltx25_pins.json" "$VARIANT" "$OPTIONAL" <<'EOF'
import json, sys
pins = json.load(open(sys.argv[1])); variant = sys.argv[2]; opt = sys.argv[3] == '1'
if variant not in ('all', *pins['variants']):
    sys.exit(f'unknown variant {variant}')
for rel, rec in pins['weights']['files'].items():
    if (variant == 'all' or variant in rec['variants']) and (rec['required'] or opt):
        print(rel)
EOF
)
REPO=$("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['weights']['hf_repo'])" "$HERE/ltx25_pins.json")
REV=$("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['weights']['hf_revision'])" "$HERE/ltx25_pins.json")
NEED_GB=$("$PY" -I - "$HERE/ltx25_pins.json" "${FILES[@]}" <<'EOF'
import json, sys
pins = json.load(open(sys.argv[1]))
print(int(sum(pins['weights']['files'][f]['bytes'] for f in sys.argv[2:]) / 1e9) + 10)
EOF
)
echo "repo $REPO revision $REV variant $VARIANT -> $DEST (${#FILES[@]} files)"

# access check: HEAD of the first gated file (prints the HTTP status only)
"$PY" -I - "$REPO" "$REV" "${FILES[0]}" <<'EOF'
import sys
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from huggingface_hub.utils import HfHubHTTPError
repo, rev, rel = sys.argv[1:4]
try:
    get_hf_file_metadata(hf_hub_url(repo, rel, revision=rev))
    print('access OK')
except HfHubHTTPError as e:
    code = e.response.status_code if e.response is not None else None
    print(f'access DENIED (HTTP {code}): accept the terms at https://huggingface.co/{repo} with the account whose access token '
          f'this server uses, then rerun', file=sys.stderr)
    sys.exit(2)
EOF

free_gb=$(df -BG --output=avail "$DEST" | tail -1 | tr -dc 0-9)
echo "free disk at $DEST: ${free_gb} GB (need about ${NEED_GB} GB)"
[[ "$free_gb" -ge "$NEED_GB" ]] || { echo "not enough disk" >&2; exit 1; }

"$ENV/venv/bin/hf" download "$REPO" "${FILES[@]}" --revision "$REV" --local-dir "$DEST"

# verify every pinned file present in DEST (this download's and any earlier one's)
"$PY" -I - "$HERE/ltx25_pins.json" "$DEST" <<'EOF'
import hashlib, json, sys, time
from pathlib import Path
pins = json.load(open(sys.argv[1])); dest = Path(sys.argv[2])
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()
rows, bad = [], []
for rel, rec in pins['weights']['files'].items():
    p = dest / rel
    if not p.is_file():
        continue
    t = time.time(); got = sha(p); size = p.stat().st_size
    ok = got == rec['sha256'] and size == rec['bytes']
    rows.append(dict(path=rel, bytes=size, sha256=got, pinned_sha256=rec['sha256'], ok=ok, variants=rec['variants'], hash_seconds=round(time.time() - t, 1)))
    if not ok:
        bad.append(rel)
    print(('OK  ' if ok else 'BAD ') + f'{got}  {rel}', flush=True)
have = {r['path'] for r in rows if r['ok']}
complete = {v: all(rel in have for rel, rec in pins['weights']['files'].items() if v in rec['variants'] and rec['required']) for v in pins['variants']}
(dest / 'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in rows))
(dest / 'weights-manifest.json').write_text(json.dumps(dict(repo=pins['weights']['hf_repo'], revision=pins['weights']['hf_revision'], files=rows,
                                                            all_ok=not bad, variants_complete=complete,
                                                            created=time.strftime('%Y-%m-%dT%H:%M:%S%z')), indent=1) + '\n')
print(json.dumps(dict(all_ok=not bad, variants_complete=complete)))
sys.exit(1 if bad else 0)
EOF
echo "weights verified: $DEST/SHA256SUMS"
