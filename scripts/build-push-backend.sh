#!/usr/bin/env bash
# Build and push the Patra backend API image to plalelab/patra-backend:latest.
# Run from repo root. Requires docker login to push.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${IMAGE:-plalelab/patra-backend:latest}"
echo "Building $IMAGE ..."
docker build -f rest_server/Dockerfile -t "$IMAGE" .
echo "Pushing $IMAGE ..."
docker push "$IMAGE"
echo "Done: $IMAGE"
