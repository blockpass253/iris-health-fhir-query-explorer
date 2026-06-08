#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  cat <<'EOF' >&2
Usage:
  scripts/publish-image.sh <ghcr-owner> <version-tag> [local-image]

Example:
  scripts/publish-image.sh pos3 v0.1.0
EOF
  exit 1
fi

owner="$1"
version_tag="$2"
local_image="${3:-iris-fhir-query-explorer-snapshot:local}"

repo="$(echo "${owner}" | tr '[:upper:]' '[:lower:]')/iris-health-fhir-query-explorer-iris"
version_image="ghcr.io/${repo}:${version_tag}"
latest_image="ghcr.io/${repo}:latest"

echo "Checking local image '${local_image}'..."
docker image inspect "${local_image}" >/dev/null

echo "Tagging images..."
docker tag "${local_image}" "${version_image}"
docker tag "${local_image}" "${latest_image}"

cat <<EOF

Ready to publish:
  ${version_image}
  ${latest_image}

If you are not logged in yet, run:
  echo <YOUR_GITHUB_TOKEN> | docker login ghcr.io -u ${owner} --password-stdin

Pushing...
EOF

docker push "${version_image}"
docker push "${latest_image}"

cat <<EOF

Publish complete.

Pull with:
  docker pull ${version_image}
  docker pull ${latest_image}
EOF
