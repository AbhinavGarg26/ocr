#!/usr/bin/env bash
# Deploy the OCR API from this Git checkout.
# Run on the server: cd /opt/ocr-api && ./deploy.sh

set -Eeuo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-ocr-api}"
IMAGE_NAME="${IMAGE_NAME:-ocr-api:latest}"
HOST_PORT="${HOST_PORT:-8001}"
CONTAINER_PORT="${CONTAINER_PORT:-8000}"
DEPLOY_BRANCH="${DEPLOY_BRANCH:-$(git -C "$APP_DIR" branch --show-current)}"

if [[ -z "$DEPLOY_BRANCH" ]]; then
  echo "Unable to determine the Git branch. Set DEPLOY_BRANCH and retry." >&2
  exit 1
fi

cd "$APP_DIR"

echo "Pulling origin/$DEPLOY_BRANCH..."
git pull --ff-only origin "$DEPLOY_BRANCH"

mkdir -p uploads

echo "Building $IMAGE_NAME..."
# Build first, so an existing working container is left untouched if the build
# fails due to a dependency or source error.
sudo docker build -f Dockerfile.vercel -t "$IMAGE_NAME" .

echo "Replacing $CONTAINER_NAME..."
sudo docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
sudo docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --cpus=1.0 \
  --memory=1g \
  --memory-reservation=512m \
  -e PORT="$CONTAINER_PORT" \
  -e OCR_MAX_IMAGE_SIDE="${OCR_MAX_IMAGE_SIDE:-2000}" \
  -e OMP_THREAD_LIMIT="${OMP_THREAD_LIMIT:-1}" \
  -e DEBUG_UPLOAD_DIR=/app/uploads \
  -v "$APP_DIR/uploads:/app/uploads" \
  -p "$HOST_PORT:$CONTAINER_PORT" \
  "$IMAGE_NAME" >/dev/null

echo "Waiting for health check..."
for attempt in {1..15}; do
  if curl -fsS "http://127.0.0.1:$HOST_PORT/health"; then
    echo
    echo "Deployment complete: http://127.0.0.1:$HOST_PORT/health"
    exit 0
  fi
  sleep 2
done

echo "Health check failed. Recent container logs:" >&2
sudo docker logs --tail 100 "$CONTAINER_NAME" >&2 || true
exit 1
