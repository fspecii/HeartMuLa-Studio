# PyInstaller hook for heartlib
# This ensures all necessary heartlib components are included

from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

datas, binaries, hiddenimports = collect_all('heartlib')

# Ensure all heartlib submodules are included
hiddenimports += collect_submodules('heartlib')

# Include transformers and torch dependencies
hiddenimports += [
    'transformers',
    'transformers.models',
    'transformers.models.auto',
    'torch',
    'torch.nn',
    'torch.optim',
    'torchaudio',
]
