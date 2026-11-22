# Sourced by the local Epona scripts (run_dev_local.sh, smoke_dev_local.sh, smoke_dev_followup.sh, profile_local.sh).
# Machine-specific settings live in the git-ignored models/epona/local.env (template local.env.example; WCB_LOCAL_ENV=FILE
# points elsewhere). The file is parsed, never executed: KEY=VALUE lines, optional 'export ' and quotes, '#' starts a
# comment. A variable already set in the environment wins; the first line for a key counts (as profile_epona.py's
# local_setting). The parser is the one of models/ltx25/local_env.sh.
load_local_env() {
  local f="${1:-${WCB_LOCAL_ENV:-$(dirname "${BASH_SOURCE[0]}")/local.env}}" line key val
  [ -f "$f" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line#export }"
    [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*)$ ]] || continue
    key="${BASH_REMATCH[1]}"
    val="${BASH_REMATCH[2]}"
    val="${val%"${val##*[![:space:]]}"}"
    val="${val#\"}"; val="${val%\"}"; val="${val#\'}"; val="${val%\'}"
    [ -n "${!key:-}" ] || export "$key=$val"
  done < "$f"
}

need_setting() {  # need_setting VAR ...: stop when a machine-specific setting is missing
  local v
  for v in "$@"; do
    [ -n "${!v:-}" ] || { echo "$v is not set: put it into models/epona/local.env (template local.env.example) or the environment" >&2; exit 2; }
  done
}

epona_settings() {  # epona_settings [EXTRA_VAR ...]: load the settings, require them, set the variables the scripts use
  load_local_env
  need_setting EPONA_ENV EPONA_MODELS_DIR WCB_GPU_LOCK WCB_LOCAL_GPU WCB_LOCAL_GPU_UUID "$@"
  local v
  for v in EPONA_ENV EPONA_SRC EPONA_MODELS_DIR WCB_GPU_LOCK WCB_LOCAL_GPU WCB_LOCAL_GPU_UUID "$@"; do
    case "${!v:-}" in *REPLACE*|/path/to/*) echo "$v still holds the placeholder of local.env.example: set it in models/epona/local.env" >&2; exit 2;; esac
  done
  PY=$EPONA_ENV/venv/bin/python                 # the locked environment (requirements-epona.lock.txt)
  SRC=${EPONA_SRC:-$EPONA_ENV/src-git}          # Epona checkout at the pinned commit (epona_pins.json)
  MODELS=$EPONA_MODELS_DIR                      # the pinned weights
  LOCK=$WCB_GPU_LOCK                            # flock file that serializes GPU jobs on this server
  GPU=$WCB_LOCAL_GPU                            # the GPU index the server's rules allow
  GPU_UUID=$WCB_LOCAL_GPU_UUID                  # its UUID: guarded_run.py and the adapter refuse any other GPU
}
