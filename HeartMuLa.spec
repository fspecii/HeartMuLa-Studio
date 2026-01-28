# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for HeartMuLa Studio macOS App

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Get the current directory
block_cipher = None
app_name = 'HeartMuLa'

# Collect data files from various packages
datas = []
datas += collect_data_files('fastapi')
datas += collect_data_files('sqlmodel')
datas += collect_data_files('heartlib', include_py_files=True)
datas += collect_data_files('transformers')
datas += collect_data_files('tokenizers')

# Include the backend directory with all Python files
datas += [('backend', 'backend')]

# Include the frontend build output
datas += [('frontend/dist', 'frontend/dist')]

# Include the models directory structure (empty placeholders)
# These will be replaced by symlinks to user directories at runtime by launcher.py
datas += [('backend/models', 'backend/models')]
datas += [('backend/generated_audio', 'backend/generated_audio')]
datas += [('backend/ref_audio', 'backend/ref_audio')]

# Collect all submodules
hiddenimports = []
hiddenimports += collect_submodules('fastapi')
hiddenimports += collect_submodules('uvicorn')
hiddenimports += collect_submodules('sqlmodel')
hiddenimports += collect_submodules('heartlib')
hiddenimports += collect_submodules('transformers')
hiddenimports += collect_submodules('torch')
hiddenimports += collect_submodules('torchaudio')
hiddenimports += ['uvicorn.logging', 'uvicorn.loops', 'uvicorn.loops.auto', 'uvicorn.protocols', 
                  'uvicorn.protocols.http', 'uvicorn.protocols.http.auto', 'uvicorn.protocols.websockets',
                  'uvicorn.protocols.websockets.auto', 'uvicorn.lifespan', 'uvicorn.lifespan.on',
                  'backend.app.main', 'backend.app.models', 'backend.app.services.music_service',
                  'backend.app.services.llm_service']

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['build/macos/hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=f'{app_name}_bin',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Disabled for macOS - UPX can cause issues with code signing
    console=False,  # No console window for cleaner UI experience. Check ~/Library/Logs/HeartMuLa/ for logs
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,  # Disabled for macOS - UPX can cause issues with code signing
    upx_exclude=[],
    name=app_name,
)

app = BUNDLE(
    coll,
    name=f'{app_name}.app',
    icon='build/macos/HeartMuLa.icns',
    bundle_identifier='com.audiohacking.heartmula',
    version='0.1.0',
    info_plist={
        'CFBundleName': 'HeartMuLa Studio',
        'CFBundleDisplayName': 'HeartMuLa Studio',
        'CFBundleExecutable': app_name,
        'CFBundleIdentifier': 'com.audiohacking.heartmula',
        'CFBundleVersion': '0.1.0',
        'CFBundleShortVersionString': '0.1.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.13',
        'NSRequiresAquaSystemAppearance': False,
        'NSAppTransportSecurity': {
            'NSAllowsArbitraryLoads': True
        },
        'NSMicrophoneUsageDescription': 'HeartMuLa Studio needs access to the microphone for audio input.',
        'LSApplicationCategoryType': 'public.app-category.music',
    },
)
