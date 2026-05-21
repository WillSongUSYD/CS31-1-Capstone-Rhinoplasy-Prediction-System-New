#!/bin/bash
# Build script for CS31-1-Rhinoplasty-Prediction-Studio-Mac.app
# Run from the repo root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "== CS31-1-Rhinoplasty-Prediction-Studio build =="
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

# 3. Clean previous build artifacts only (leave other files in dist/ intact).
rm -rf build
rm -rf dist/CS31-1-Rhinoplasty-Prediction-Studio-Mac.app
rm -f  dist/CS31-1-Rhinoplasty-Prediction-Studio-Mac.zip
mkdir -p dist

# 4. Build.
echo "== py2app build =="
.venv/bin/python desktop/setup.py py2app --arch arm64 2>&1 | tail -30

# 5a. Fix torchvision rpath.
# torchvision/_C.so + image.so are linked with @rpath/libc10.dylib but
# their embedded rpath points at the PyTorch build CI machine
# (/Users/ec2-user/runner/...). After py2app copies them verbatim into
# the bundle the rpath is meaningless and dlopen fails silently, which
# manifests as 'operator torchvision::nms does not exist' at runtime.
# Inject a relative rpath so they find libc10.dylib inside our bundle.
echo "== fix torchvision rpath =="
PYVER=$(.venv/bin/python -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
VISION_DIR="dist/CS31-1-Rhinoplasty-Prediction-Studio-Mac.app/Contents/Resources/lib/${PYVER}/torchvision"
for f in "$VISION_DIR/_C.so" "$VISION_DIR/image.so"; do
    if [ -f "$f" ]; then
        # Idempotent: silently skip if already fixed.
        if ! otool -l "$f" 2>/dev/null | grep -q "@loader_path/../torch/lib"; then
            install_name_tool -add_rpath @loader_path/../torch/lib "$f" 2>&1 \
                | grep -v "warning: changes being made to the file will invalidate the code signature" || true
            # install_name_tool invalidates the pre-existing ad-hoc
            # signature. macOS 26+ enforces signatures strictly: an
            # invalid signature causes dyld to kill the process with
            # SIGKILL + "Code Signature Invalid" at dlopen time —
            # which looks exactly like a silent app crash. Re-sign
            # ad-hoc so the signature is valid again.
            codesign --force --sign - "$f" 2>&1 | grep -v "replacing existing" || true
            echo "  patched + re-signed $f"
        else
            echo "  $f already patched"
        fi
    fi
done

# 5b. Re-sign the entire bundle after rpath patching.
# Individual install_name_tool calls above invalidate the bundle-level
# signature; without this step macOS shows "damaged" instead of "unverified",
# and users cannot use the Privacy & Security → Open Anyway workaround.
echo "== re-sign bundle =="
codesign --force --deep --sign - dist/CS31-1-Rhinoplasty-Prediction-Studio-Mac.app
echo "  bundle re-signed (ad-hoc)"

# 5c. Verify the bundle.
echo "== verify =="
.venv/bin/python desktop/scripts/verify_bundle.py dist/CS31-1-Rhinoplasty-Prediction-Studio-Mac.app

# 6. Zip for distribution (ditto preserves macOS-specific metadata
# like code-signing attributes; plain zip strips those).
echo "== zip =="
rm -f dist/CS31-1-Rhinoplasty-Prediction-Studio-Mac.zip
(cd dist && ditto -c -k --sequesterRsrc --keepParent CS31-1-Rhinoplasty-Prediction-Studio-Mac.app CS31-1-Rhinoplasty-Prediction-Studio-Mac.zip)

echo
echo "== done =="
du -sh dist/CS31-1-Rhinoplasty-Prediction-Studio-Mac.app dist/CS31-1-Rhinoplasty-Prediction-Studio-Mac.zip
echo
echo "Distribute dist/CS31-1-Rhinoplasty-Prediction-Studio-Mac.zip."
echo "On first launch, users must right-click the app → Open → Open"
echo "to bypass Gatekeeper (unsigned)."
