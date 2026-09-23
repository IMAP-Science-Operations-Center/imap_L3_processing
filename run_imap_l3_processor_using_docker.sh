#!/bin/bash

set -e

if [[ -z "$IMAP_API_KEY" ]]; then
    echo "Warning: IMAP_API_KEY not set; you might get errors."
fi

if [[ -z "$IMAP_DATA_DIR" ]]; then
    IMAP_DATA_DIR="$PWD/data"
    mkdir -p "$IMAP_DATA_DIR"
    echo "Warning: IMAP_DATA_DIR not set; using default directory ('$IMAP_DATA_DIR')"
fi

if ! docker image inspect imap-l3 >/dev/null 2>&1; then
    echo "Docker image 'imap-l3' not found; building it..."
    docker build --platform=linux/amd64 -t imap-l3 .
    echo "Done!"
fi

echo "Running 'imap-l3' Docker image..."
docker run --platform=linux/amd64 --rm \
    -e IMAP_API_KEY \
    -e IMAP_DATA_DIR=/data \
    -v "$IMAP_DATA_DIR:/data" \
    imap-l3 "$@"
echo "Done!"
