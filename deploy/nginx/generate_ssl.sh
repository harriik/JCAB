#!/usr/bin/env bash
# Generates self-signed SSL certificates for local testing & HTTPS microphone access
set -euo pipefail

SSL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ssl"
mkdir -p "${SSL_DIR}"

if [[ -f "${SSL_DIR}/server.crt" && -f "${SSL_DIR}/server.key" ]]; then
    echo "[SSL] Existing certificates found in ${SSL_DIR}."
    exit 0
fi

echo "[SSL] Generating self-signed SSL certificate in ${SSL_DIR}..."
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout "${SSL_DIR}/server.key" \
    -out "${SSL_DIR}/server.crt" \
    -subj "/C=US/ST=State/L=City/O=AdaptiveAssessment/OU=Dev/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:0.0.0.0"

chmod 600 "${SSL_DIR}/server.key"
chmod 644 "${SSL_DIR}/server.crt"
echo "[SSL] Certificate generation complete."
