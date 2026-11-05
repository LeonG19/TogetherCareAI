#!/usr/bin/env bash
# Start/stop/inspect the TogetherCare stack (vLLM + backend + cloudflared).
#
#   ./scripts/services.sh start|stop|restart|status|url|share|logs|install
#
# All three run as systemd --user units with Restart=always, and lingering is
# enabled for this account, so they survive logout and reboot. They are NOT
# tied to the SSH session that started them.
#
# Ports on this host:
#   8001  vLLM            127.0.0.1 ONLY - never exposed, never tunnelled
#   8010  backend         127.0.0.1 ONLY - reached only via the web tier
#   8080  web tier        127.0.0.1, THE tunnelled one (app + /api + gate)
#   8000  ANOTHER USER'S uvicorn - do not touch, do not tunnel
#   8002  ANOTHER USER'S service  - do not touch

set -uo pipefail

# Start order matters only loosely (every unit tolerates its dependency being
# down), but listing the web tier before the tunnel keeps `status` readable.
UNITS=(togethercare-vllm togethercare-backend togethercare-web togethercare-tunnel)
LOGDIR=/mnt/lagarza3/logs
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

tunnel_url() {
  # A named tunnel (fixed hostname, set up by scripts/setup_named_tunnel.sh)
  # wins if one is configured. Otherwise fall back to scraping the quick
  # tunnel's random hostname out of the log, which is where it only ever
  # appears - and which changes on every restart.
  local h
  h="$(grep -E '^[[:space:]]+- hostname:' "$HOME/.cloudflared/config.yml" 2>/dev/null \
       | head -1 | awk '{print $3}')"
  if [[ -n "$h" ]]; then
    echo "https://$h"
    return
  fi
  grep -aoE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOGDIR/tunnel.log" 2>/dev/null | tail -1
}

case "${1:-status}" in
  install)
    # Re-link the units from a fresh clone. Safe to re-run.
    mkdir -p "$HOME/.config/systemd/user" "$LOGDIR"
    loginctl enable-linger "$USER" || echo "WARN: could not enable lingering; units will die at logout"
    systemctl --user daemon-reload
    systemctl --user enable "${UNITS[@]}"
    echo "installed. now: $0 start"
    ;;

  start)
    systemctl --user start "${UNITS[@]}"
    echo "starting. vLLM needs ~2 min to load 16 GB of weights before it binds :8001."
    echo "watch with: $0 logs"
    ;;

  stop)
    # Only ever touches this user's own units.
    systemctl --user stop "${UNITS[@]}"
    echo "stopped."
    ;;

  restart)
    systemctl --user restart "${UNITS[@]}"
    echo "restarted. NOTE: the quick tunnel gets a NEW public URL every restart."
    echo "get it with: $0 url"
    ;;

  status)
    systemctl --user list-units 'togethercare-*' --no-pager --no-legend
    echo
    # Different paths per port: vLLM serves its own /health, the backend's
    # equivalent was renamed to /togethercareAI, and the web tier answers
    # /togethercareAI/web without needing the access code.
    for pp in "8001:/health" "8010:/togethercareAI" "8080:/togethercareAI/web"; do
      p="${pp%%:*}"; path="${pp#*:}"
      printf '  :%s  ' "$p"
      curl -s -o /dev/null --max-time 3 "http://127.0.0.1:$p$path" \
        && echo "up" || echo "not answering (vLLM takes ~2 min to load)"
    done
    echo
    echo "  tunnel: $(tunnel_url || echo '<none yet>')"
    ;;

  url)
    u="$(tunnel_url)"
    if [[ -z "$u" ]]; then
      echo "no tunnel URL in $LOGDIR/tunnel.log - is togethercare-tunnel running?" >&2
      exit 1
    fi
    echo "$u"
    ;;

  share)
    # Everything a tester needs, in one place. A quick tunnel gets a NEW
    # hostname on every restart, so this is the command to re-run before
    # sending the link out again.
    u="$(tunnel_url)"
    if [[ -z "$u" ]]; then
      echo "no tunnel URL yet - is togethercare-tunnel running?" >&2
      exit 1
    fi
    code="$(grep '^ACCESS_CODE=' "$SRC/web/.env" 2>/dev/null | head -1 | cut -d= -f2-)"
    echo "  URL:  $u"
    echo "  Code: ${code:-<not set - see web/.env>}"
    ;;

  logs)
    echo "tailing $LOGDIR/{vllm,backend,web,tunnel}.log - Ctrl-C to stop"
    tail -n 20 -f "$LOGDIR"/vllm.log "$LOGDIR"/backend.log \
                 "$LOGDIR"/web.log "$LOGDIR"/tunnel.log
    ;;

  *)
    echo "usage: $0 start|stop|restart|status|url|share|logs|install" >&2
    exit 2
    ;;
esac
