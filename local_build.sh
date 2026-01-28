#!/bin/bash
# ---------------------------------------------------------------------------
#  HeartMuLa Studio - Local Build Script
#  Replicates the GitHub Actions build process for local testing
# ---------------------------------------------------------------------------

set -e  # Exit on error

echo "=========================================="
echo "HeartMuLa Studio - Local Build"
echo "=========================================="
echo ""

# This script mirrors .github/workflows/build-macos-release.yml
# for local testing without needing to push to GitHub

cd "$(dirname "$0")"

# Check Python 3.11 (same as GitHub Actions)
if ! command -v python3.11 &> /dev/null && ! python3 --version 2>&1 | grep -q "3.11"; then
    echo "ERROR: Python 3.11 not found"
    echo "GitHub Actions uses Python 3.11. Please install it for consistent builds."
    exit 1
fi

PYTHON="python3"
if command -v python3.11 &> /dev/null; then
    PYTHON="python3.11"
fi

echo "Using Python: $($PYTHON --version)"
echo ""

# Install system dependencies for icon generation
echo "Checking system dependencies..."
if ! command -v rsvg-convert &> /dev/null && ! command -v convert &> /dev/null; then
    echo "WARNING: Icon generation tools not found"
    echo "Install with: brew install librsvg imagemagick"
    echo ""
fi

# Install Python dependencies
echo "Installing Python dependencies..."
$PYTHON -m pip install --upgrade pip
pip install -r requirements_macos.txt

echo ""

# Build frontend
echo "Building frontend..."
cd frontend
npm install
npm run build
cd ..

echo ""

# Generate app icon
echo "Generating app icon..."
chmod +x build/macos/generate_icon.sh
./build/macos/generate_icon.sh

echo ""

# Check build/macos assets (same checks as GitHub Actions)
echo "Checking build assets..."
if [ ! -f "build/macos/HeartMuLa.icns" ]; then
    echo "ERROR: build/macos/HeartMuLa.icns not found after generation."
    exit 1
fi
if [ ! -f "build/macos/codesign.sh" ]; then
    echo "ERROR: build/macos/codesign.sh not found."
    exit 1
fi
if [ ! -f ".github/DMG_README.txt" ]; then
    echo "ERROR: .github/DMG_README.txt not found."
    exit 1
fi

echo "✓ All build assets present"
echo ""

# Clean previous PyInstaller outputs
echo "Cleaning previous builds..."
rm -rf dist/HeartMuLa.app build/HeartMuLa

# Build with PyInstaller
echo "Building with PyInstaller..."
$PYTHON -m PyInstaller HeartMuLa.spec --clean --noconfirm

echo ""

# Set up app bundle executable
echo "Setting up app bundle executable..."
cp dist/HeartMuLa.app/Contents/MacOS/HeartMuLa_bin dist/HeartMuLa.app/Contents/MacOS/HeartMuLa
chmod +x dist/HeartMuLa.app/Contents/MacOS/HeartMuLa

# Code sign the app bundle
echo ""
echo "Code signing app bundle..."
chmod +x build/macos/codesign.sh
MACOS_SIGNING_IDENTITY="-" ./build/macos/codesign.sh dist/HeartMuLa.app

echo ""
echo "=========================================="
echo "✓ Build Complete!"
echo "=========================================="
echo ""
echo "App: dist/HeartMuLa.app"
echo ""
echo "To test:"
echo "  open dist/HeartMuLa.app"
echo ""
echo "If blocked by Gatekeeper:"
echo "  xattr -cr dist/HeartMuLa.app"
echo "  (then right-click → Open)"
echo ""
