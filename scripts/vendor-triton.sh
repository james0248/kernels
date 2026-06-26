#!/usr/bin/env bash
# Vendor Triton's Python source into .triton-src/ for Mac autocomplete (never executed).
set -euo pipefail
cd "$(dirname "$0")/.."

ver="$(grep -A1 '^name = "triton"$' uv.lock | sed -nE 's/^version = "([^"]+)".*/\1/p' | head -1)"
[ -n "$ver" ] || { echo "no triton version in uv.lock; run 'uv lock'" >&2; exit 1; }

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
git clone --depth 1 --branch "v$ver" --filter=blob:none --sparse \
  https://github.com/triton-lang/triton "$tmp" >/dev/null 2>&1
git -C "$tmp" sparse-checkout set python/triton >/dev/null 2>&1
rm -rf .triton-src/triton && mkdir -p .triton-src
cp -R "$tmp/python/triton" .triton-src/triton
echo "vendored triton v$ver -> .triton-src/ (restart language server)"
