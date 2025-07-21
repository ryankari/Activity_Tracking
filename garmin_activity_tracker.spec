# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('garmin_activity_tracker/prompt_template.txt', 'garmin_activity_tracker'),
        # Add other data files here if needed, e.g.:
        # ('data/garminSummaryData.xlsx', 'data'),
        # ('data/garminSplitData.xlsx', 'data'),
    ],
    hiddenimports=[
        'jinja2', 'PyQt5', 'matplotlib', 'pandas', 'numpy'
        # Add any other hidden imports your app needs
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='garmin-activity-tracker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Set to True if you want a console window
    icon=None,      # You can specify an .ico file here if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='garmin-activity-tracker'
)