# Sourced by the local Wan2.1-VACE-1.3B shell tools (run_local_chunks.sh, run_probe.sh). Machine-specific settings live in
# the git-ignored models/wan21_vace13b/local.env (template local.env.example; WCB_LOCAL_ENV=FILE points elsewhere). The
# file is parsed, never executed: KEY=VALUE lines, optional 'export ' and quotes, '#' starts a comment. A variable already
# set in the environment wins; the first line for a key counts. The parser is the one of models/vista/local_env.sh (and of
# models/ltx25 and models/epona); vace_common.py implements the same rules for the Python tools.
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
    [ -n "${!v:-}" ] || { echo "$v is not set: put it into models/wan21_vace13b/local.env (template local.env.example) or the environment" >&2; exit 2; }
    case "${!v}" in *REPLACE*|/path/to/*) echo "$v still holds the placeholder of local.env.example: set it in models/wan21_vace13b/local.env" >&2; exit 2;; esac
  done
}
