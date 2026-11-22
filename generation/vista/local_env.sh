# Sourced by the local Vista scripts (run_dev_smoke.sh). Machine-specific settings live in the git-ignored
# models/vista/local.env (template local.env.example; WCB_LOCAL_ENV=FILE points elsewhere). The file is parsed, never
# executed: KEY=VALUE lines, optional 'export ' and quotes, '#' starts a comment. A variable already set in the environment
# wins; the first line for a key counts. The parser is the one of models/ltx25/local_env.sh and models/epona/local_env.sh.
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

need_setting() {  # need_setting VAR ...: stop when a machine-specific setting is missing or still a placeholder
  local v
  for v in "$@"; do
    [ -n "${!v:-}" ] || { echo "$v is not set: put it into models/vista/local.env (template local.env.example) or the environment" >&2; exit 2; }
    case "${!v}" in *REPLACE*|/path/to/*) echo "$v still holds the placeholder of local.env.example: set it in models/vista/local.env" >&2; exit 2;; esac
  done
}

vista_settings() {  # vista_settings [EXTRA_VAR ...]: load the settings, require them, set the variables the scripts use
  load_local_env
  need_setting VISTA_ENV VISTA_CKPT WCB_OUTPUTS WCB_INSTRUMENT_PY WCB_SCRIPTS WCB_SNAPSHOT_DIR WCB_GPU_LOCK WCB_LOCAL_GPU WCB_LOCAL_GPU_UUID "$@"
  PY=$VISTA_ENV/venv/bin/python               # the locked environment (requirements-vista.lock.txt)
  SRC=${VISTA_SRC:-$VISTA_ENV/src-git}        # Vista checkout at the pinned commit (vista_pins.json)
  CKPT=$VISTA_CKPT                            # the pinned weights
  IPY=$WCB_INSTRUMENT_PY                      # the bench instrument's Python (detection, band odometry)
  LOCK=$WCB_GPU_LOCK                          # flock file that serializes GPU jobs on this server
  GPU=$WCB_LOCAL_GPU                          # the GPU index the server's rules allow
  GPU_UUID=$WCB_LOCAL_GPU_UUID                # its UUID: guarded_run.py and the adapter refuse any other GPU
}
