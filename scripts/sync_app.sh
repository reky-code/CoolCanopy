#!/usr/bin/env sh
# Copy the shared heat package into the app folder (Databricks Apps upload only app/).
set -e
root="$(cd "$(dirname "$0")/.." && pwd)"
rm -rf "$root/app/heat"
cp -R "$root/src/heat" "$root/app/heat"
find "$root/app/heat" -name __pycache__ -type d -prune -exec rm -rf {} +
echo "synced src/heat -> app/heat"
