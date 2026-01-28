#!/bin/bash
# Code signing script for HeartMuLa Studio macOS app
# This script signs the app bundle to prevent "app is damaged" warnings

set -e

APP_PATH="$1"
SIGNING_IDENTITY="${MACOS_SIGNING_IDENTITY:--}"

if [ -z "$APP_PATH" ]; then
    echo "Usage: $0 <path-to-app>"
    exit 1
fi

if [ ! -d "$APP_PATH" ]; then
    echo "Error: App bundle not found at $APP_PATH"
    exit 1
fi

echo "Code signing $APP_PATH..."
echo "Signing identity: $SIGNING_IDENTITY"

# Function to sign a file or directory
sign_item() {
    local item="$1"
    echo "  Signing: $item"
    
    # For ad-hoc signing, don't use runtime option
    if [ "$SIGNING_IDENTITY" = "-" ]; then
        codesign --force --deep --sign "$SIGNING_IDENTITY" "$item"
    else
        # For real certificates, use runtime hardening with fallback
        codesign --force --deep --sign "$SIGNING_IDENTITY" \
            --options runtime \
            --timestamp \
            "$item" 2>/dev/null || \
        codesign --force --deep --sign "$SIGNING_IDENTITY" \
            "$item"
    fi
}

# Sign all dylibs and frameworks first (inside-out signing)
find "$APP_PATH/Contents" -type f \( -name "*.dylib" -o -name "*.so" \) -print0 | while IFS= read -r -d '' file; do
    sign_item "$file"
done

# Sign frameworks
find "$APP_PATH/Contents/Frameworks" -type d -name "*.framework" 2>/dev/null | while read -r framework; do
    sign_item "$framework"
done

# Sign the main executable
if [ -f "$APP_PATH/Contents/MacOS/HeartMuLa" ]; then
    sign_item "$APP_PATH/Contents/MacOS/HeartMuLa"
fi

if [ -f "$APP_PATH/Contents/MacOS/HeartMuLa_bin" ]; then
    sign_item "$APP_PATH/Contents/MacOS/HeartMuLa_bin"
fi

# Finally, sign the entire app bundle
sign_item "$APP_PATH"

echo "Code signing complete!"

# Verify the signature
echo ""
echo "Verifying signature..."
codesign --verify --deep --verbose=2 "$APP_PATH" 2>&1 | head -5
echo ""
echo "✓ Code signing verification passed"
