#!/usr/bin/env bash
set -euo pipefail

container_name="${1:-fhir-template}"
snapshot_image="${2:-iris-fhir-query-explorer-snapshot:local}"

echo "Checking container '${container_name}'..."
docker inspect "${container_name}" >/dev/null

echo "Creating snapshot image '${snapshot_image}' from '${container_name}'..."
docker commit "${container_name}" "${snapshot_image}" >/dev/null

echo "Snapshot complete."
echo "Use it with compose locally or publish it with:"
echo "  scripts/publish-image.sh <github-owner> <version-tag> ${snapshot_image}"
