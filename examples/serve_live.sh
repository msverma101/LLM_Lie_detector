#!/bin/bash
# Start the web UI as a user service and expose it through a temporary Cloudflare quick tunnel.
#   - Access: bearer token / cookie (random, written to runs/ui/token, mode 0600); 12 scoring requests per minute per client IP.
#   - The tunnel URL is random and temporary; it ends when the unit is stopped or the machine reboots.
#   - A private ~/.cloudflared/config.yml would be inherited by a quick tunnel (its catch-all 404 breaks it),
#     so an EMPTY config file is passed explicitly. Your own config is never touched.
# Stop:   systemctl --user stop liedet-tunnel liedet-ui
# Rotate the token: restart liedet-ui (a new token is generated and written to runs/ui/token).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${QWEN_DIR:?set QWEN_DIR to the Qwen3.5-2B snapshot directory}
RUN=${RUN_DIR:-runs/20261107-main}
RESULTS=${RESULTS_DIR:-docs/data/experiments/fibs-wsp-2026-11-07}
mkdir -p runs/ui; : > runs/ui/cloudflared-empty.yml
systemctl --user stop liedet-tunnel liedet-ui 2>/dev/null || true
systemd-run --user --unit=liedet-ui --collect --working-directory="$PWD" --property=Restart=on-failure --property=MemoryMax=9G \
  --setenv=HF_HUB_OFFLINE=1 --setenv=PYTHONUNBUFFERED=1 "$PWD/.venv/bin/python" -m liedet serve --host 127.0.0.1 --port 8790 \
  --results "$RESULTS" --model "$MODEL" --run "$RUN" --token-file runs/ui/token --per-minute 12
until curl -fs -m 3 http://127.0.0.1:8790/api/health >/dev/null; do sleep 3; done
INV=$(systemd-run --user --unit=liedet-tunnel --collect --property=Restart=on-failure /usr/local/bin/cloudflared \
  --config "$PWD/runs/ui/cloudflared-empty.yml" tunnel --no-autoupdate --url http://127.0.0.1:8790 2>&1 | grep -oE "invocation ID: [0-9a-f]+" | cut -d' ' -f3)
for _ in $(seq 1 40); do
  U=$(journalctl --user _SYSTEMD_INVOCATION_ID="$INV" --no-pager | grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" | tail -1); [ -n "$U" ] && break; sleep 2
done
echo "$U" > runs/ui/public_url.txt; chmod 600 runs/ui/public_url.txt
echo "Open:  $U/?t=$(cat runs/ui/token)"
