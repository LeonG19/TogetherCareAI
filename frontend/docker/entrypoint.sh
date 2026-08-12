#!/bin/sh
# Regenerates env-config.js from environment variables at container startup,
# so BACKEND_URL (etc.) can be changed by editing the Cloudflare Container's
# env vars and restarting the container - no image rebuild needed.
set -eu

cat > /app/dist/env-config.js <<EOF
window.__ENV = {
  BACKEND_URL: "${BACKEND_URL:-}",
  API_KEY: "${API_KEY:-}",
  AB_ENABLED: "${AB_ENABLED:-0}",
  AB_EVERY_N: "${AB_EVERY_N:-5}"
};
EOF

exec node /app/server.js
