#!/bin/bash
# Script to generate HeartMuLa.icns from SVG icon
# This will be run during the GitHub Actions build process

set -e

SVG_FILE="frontend/public/heartmula-icon.svg"
ICONSET_DIR="build/macos/HeartMuLa.iconset"
OUTPUT_ICNS="build/macos/HeartMuLa.icns"

# Check if SVG file exists
if [ ! -f "$SVG_FILE" ]; then
    echo "Error: SVG icon file not found at $SVG_FILE"
    exit 1
fi

# Check if imagemagick (convert) and iconutil are available
if ! command -v rsvg-convert &> /dev/null && ! command -v convert &> /dev/null; then
    echo "Error: Neither rsvg-convert nor ImageMagick convert found"
    echo "Install with: brew install librsvg imagemagick"
    exit 1
fi

# Create iconset directory
rm -rf "$ICONSET_DIR"
mkdir -p "$ICONSET_DIR"

echo "Generating icon sizes from SVG..."

# Generate all required icon sizes
# macOS requires specific sizes for .icns files
sizes=(16 32 128 256 512)

for size in "${sizes[@]}"; do
    # Standard resolution
    echo "  Generating ${size}x${size}..."
    if command -v rsvg-convert &> /dev/null; then
        rsvg-convert -w $size -h $size "$SVG_FILE" -o "$ICONSET_DIR/icon_${size}x${size}.png"
    else
        convert -background none -resize ${size}x${size} "$SVG_FILE" "$ICONSET_DIR/icon_${size}x${size}.png"
    fi
    
    # Retina resolution (@2x)
    double=$((size * 2))
    echo "  Generating ${size}x${size}@2x (${double}x${double})..."
    if command -v rsvg-convert &> /dev/null; then
        rsvg-convert -w $double -h $double "$SVG_FILE" -o "$ICONSET_DIR/icon_${size}x${size}@2x.png"
    else
        convert -background none -resize ${double}x${double} "$SVG_FILE" "$ICONSET_DIR/icon_${size}x${size}@2x.png"
    fi
done

# Convert iconset to icns
echo "Creating .icns file..."
iconutil -c icns "$ICONSET_DIR" -o "$OUTPUT_ICNS"

# Clean up
rm -rf "$ICONSET_DIR"

echo "✓ Icon created: $OUTPUT_ICNS"
ls -lh "$OUTPUT_ICNS"
