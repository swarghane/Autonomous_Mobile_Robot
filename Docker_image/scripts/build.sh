#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "========================================="
echo "Building AMR Base Image..."
echo "========================================="

docker build \
    --no-cache \
    -t amr_base_image:latest \
    .

echo ""
echo "========================================="
echo "Build Completed Successfully!"
echo "========================================="