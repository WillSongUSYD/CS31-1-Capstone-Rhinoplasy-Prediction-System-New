#!/bin/bash
# Build script for CS31Preview.app
# Run from the repo root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "== CS31Preview build =="
echo "repo: $REPO_ROOT"

# 1. Verify the V6 LoRA is on disk (py2app will copy it into the bundle).
LORA_PATH="$REPO_ROOT/models/outcome_v3_512/sd_inpaint_nose_v6/step_10000/pytorch_lora_weights.safetensors"
if [ ! -f "$LORA_PATH" ]; then
    echo "ERROR: V6 LoRA not found at $LORA_PATH"
    echo "       Pull it from the remote training server first."
    exit 1
fi
echo "  LoRA: $(du -h "$LORA_PATH" | cut -f1)"

# 2. Verify py2app is installed.
if ! .venv/bin/python -c "import py2app" 2>/dev/null; then
    echo "installing py2app ..."
    .venv/bin/pip install --quiet py2app
fi

# 3. Clean previous build.
rm -rf build dist

# 4. Build.
echo "== py2app build =="
.venv/bin/python desktop/setup.py py2app --arch arm64 2>&1 | tail -30

# 5. Verify the bundle.
echo "== verify =="
.venv/bin/python desktop/scripts/verify_bundle.py dist/CS31Preview.app

# 6. Zip for distribution (ditto preserves macOS-specific metadata
# like code-signing attributes; plain zip strips those).
echo "== zip =="
rm -f dist/CS31Preview.zip
cd dist && ditto -c -k --sequesterRsrc --keepParent CS31Preview.app CS31Preview.zip

echo
echo "== done =="
du -sh dist/CS31Preview.app dist/CS31Preview.zip
echo
echo "Distribute dist/CS31Preview.zip."
echo "On first launch, users must right-click the app → Open → Open"
echo "to bypass Gatekeeper (unsigned)."
