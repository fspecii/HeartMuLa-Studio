# HeartMuLa Studio - macOS Build

This directory contains assets and scripts for building HeartMuLa Studio as a standalone macOS application using PyInstaller.

## Directory Structure

```
build/macos/
├── HeartMuLa.icns           # App icon (generated from SVG)
├── codesign.sh              # Code signing script
├── generate_icon.sh         # Script to generate .icns from SVG
└── hooks/
    └── hook-heartlib.py     # PyInstaller hook for heartlib
```

## Building the macOS App

### Prerequisites

- macOS 10.13 or later
- Python 3.11
- Node.js 18+
- Homebrew (for icon generation tools)

### Automated Build Script

For the easiest local build experience, use the provided build script:

```bash
./local_build.sh
```

This script replicates the GitHub Actions build process locally.

### Validate Build Assets

Before building, you can validate that all required files are present:

```bash
./validate_build_assets.sh
```

This checks for:
- Required scripts (generate_icon.sh, codesign.sh)
- Required directories (backend/models, frontend/dist, etc.)
- Dependency configuration (transformers version constraints, triton, etc.)
- PyInstaller spec file configuration

### Manual Build

1. Install dependencies:
```bash
brew install librsvg imagemagick
pip install -r requirements_macos.txt
```

2. Build frontend:
```bash
cd frontend
npm install
npm run build
cd ..
```

3. Generate icon:
```bash
./build/macos/generate_icon.sh
```

4. Build with PyInstaller:
```bash
pyinstaller HeartMuLa.spec --clean --noconfirm
```

5. Set up executable:
```bash
cp dist/HeartMuLa.app/Contents/MacOS/HeartMuLa_bin dist/HeartMuLa.app/Contents/MacOS/HeartMuLa
chmod +x dist/HeartMuLa.app/Contents/MacOS/HeartMuLa
```

6. Code sign:
```bash
./build/macos/codesign.sh dist/HeartMuLa.app
```

7. Create DMG:
```bash
mkdir -p dmg_temp
cp -R dist/HeartMuLa.app dmg_temp/
cp HeartMuLa.command dmg_temp/
ln -s /Applications dmg_temp/Applications
cp .github/DMG_README.txt dmg_temp/README.txt
hdiutil create -volname "HeartMuLa Studio" -srcfolder dmg_temp -ov -format UDZO HeartMuLa-macOS.dmg
```

### GitHub Actions

The build process is automated via GitHub Actions. See `.github/workflows/build-macos-release.yml`.

To trigger a build:
1. Create a new release on GitHub, or
2. Manually trigger the workflow from the Actions tab

## Metal GPU Optimization

The macOS build is optimized for Apple Metal GPUs:

- PyTorch with MPS (Metal Performance Shaders) backend
- Automatic fallback for unsupported operations
- Environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` is set by default

### Apple Silicon (M1/M2/M3)

Native ARM64 support with Metal acceleration provides best performance.

### Intel Macs

Compatible with Intel Macs that have Metal-capable GPUs (2012 or later).

## Data Storage

**IMPORTANT**: The macOS app stores ALL data in the user's Library directory, NOT in the app bundle:

```
~/Library/Application Support/HeartMuLa/
├── models/              # AI models (~5GB, auto-downloaded)
├── generated_audio/     # Generated music files
├── ref_audio/           # Uploaded reference audio
└── jobs.db              # Song history database

~/Library/Logs/HeartMuLa/
└── (application logs)
```

The launcher (`launcher.py`) sets environment variables to ensure all data writes go to the user directory:
- `HEARTMULA_MODEL_DIR` → models directory
- `HEARTMULA_GENERATED_AUDIO_DIR` → generated audio directory
- `HEARTMULA_REF_AUDIO_DIR` → reference audio directory
- `HEARTMULA_DB_PATH` → database path

This approach ensures:
- The app bundle remains read-only (as required by code signing)
- User data persists across app updates
- Users can easily find and manage their generated music
- Standard macOS app behavior (data in Library folder)

## Code Signing

The build includes ad-hoc code signing by default to prevent "damaged app" warnings.

For production releases, set the `MACOS_SIGNING_IDENTITY` secret in GitHub to your Apple Developer ID.

## Troubleshooting

### Icon not generating

Install the required tools:
```bash
brew install librsvg imagemagick
```

### PyInstaller errors

Ensure all dependencies are installed:
```bash
pip install -r requirements_macos.txt
pip install pyinstaller
```

Common issues:
- **Missing backend/models directory**: Fixed in latest version (directory created with .gitkeep)
- **Transformers version conflict**: requirements_macos.txt now constrains to `<4.57.0` to avoid yanked version
- **Triton platform compatibility**: Triton is not compatible with Apple Silicon. The dependency includes a platform constraint to skip installation on ARM64 Macs.

### App won't open

Right-click the app and select "Open" to bypass Gatekeeper.

For permanent fix, code sign with a valid Developer ID certificate.
