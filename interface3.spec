# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['interface3.py'],
    pathex=[r'D:\ESI\3CS\stage\GUI'],  # Use raw string for path
    binaries=[],
    datas=[
        (r'D:\ESI\3CS\stage\GUI\CDTA Logo white.png', 'CDTA Logo white.png'),
        (r'D:\ESI\3CS\stage\GUI\AC2_Logo_NEW white.png', 'AC2_Logo_NEW white.png'),
        (r'D:\ESI\3CS\stage\GUI\esi-logo-white.png', 'esi-logo-white.png'),
        (r'D:\ESI\3CS\stage\GUI\UNet_wavelet_fusion_150epoch_model_12.h5', 'UNet_wavelet_fusion_150epoch_model_12.h5')
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='interface3',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
