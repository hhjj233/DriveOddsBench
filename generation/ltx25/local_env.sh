# Sourced by the local LTX-2.5 scripts (run_dev_smoke.sh, download_ltx25_weights.sh). Machine-specific settings live in the
# git-ignored models/ltx25/local.env (template local.env.example; WCB_LOCAL_ENV=FILE points elsewhere). The file is parsed,
# never executed: KEY=VALUE lines, optional 'export ' and quotes, '#' starts a comment. A variable already set in the
# environment wins; the first line for a key counts (as in run_ltx25_bench.py's local_setting).
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
    [ -n "${!v:-}" ] || { echo "$v is not set: put it into models/ltx25/local.env (template local.env.example) or the environment" >&2; exit 2; }
  done
}
