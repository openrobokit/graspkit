#!/bin/bash
# Build coacd-acvd Docker image and process all dex_diffuser objects.
#
# Usage:
#   bash scripts/stages/asset_prep/process_dexdiffuser_mp.sh
#   NUM_WORKERS=16 bash scripts/stages/asset_prep/process_dexdiffuser_mp.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
INPUT_DIR="$PROJECT_ROOT/data/object_data/dex_diffuser"
OUTPUT_DIR="$PROJECT_ROOT/data/object_data/dex_diffuser/processed"
SCRIPT_PATH="$PROJECT_ROOT/scripts/stages/asset_prep/process_dexdiffuser.py"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKERFILE="$SCRIPT_DIR/Dockerfile.coacd_acvd"
PATCH_FILE="$SCRIPT_DIR/coacd_python_manifold.patch"
IMAGE_NAME="coacd-acvd:latest"
NUM_WORKERS="${NUM_WORKERS:-16}"

echo "=== Dex-Diffuser Mesh Processing ==="
echo "Input:   $INPUT_DIR"
echo "Output:  $OUTPUT_DIR"
echo "Workers: $NUM_WORKERS"

mkdir -p "$OUTPUT_DIR"

# Build Docker image if not present (or REBUILD=1)
if [ "${REBUILD:-0}" = "1" ] || ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo ""
    echo "=== Building Docker image ==="
    BUILD_CONTEXT=$(mktemp -d)
    trap 'rm -rf "$BUILD_CONTEXT"' EXIT
    cp "$PATCH_FILE" "$BUILD_CONTEXT/coacd_python_manifold.patch"
    docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" "$BUILD_CONTEXT"
else
    echo "Docker image $IMAGE_NAME already exists (set REBUILD=1 to force rebuild)"
fi

echo ""
echo "=== Processing ==="
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e INPUT_DIR=/data/input \
    -e OUTPUT_DIR=/data/output \
    -e NUM_WORKERS="$NUM_WORKERS" \
    -v "$INPUT_DIR:/data/input:ro" \
    -v "$OUTPUT_DIR:/data/output" \
    -v "$SCRIPT_PATH:/workspace/process.py:ro" \
    --entrypoint python3 \
    "$IMAGE_NAME" \
    /workspace/process.py
