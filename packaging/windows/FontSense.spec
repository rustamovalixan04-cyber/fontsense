from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)


ROOT = Path(SPECPATH).resolve().parents[1]

datas = [
    (
        str(ROOT / "artifacts" / "cnn" / "cnn_model.pt"),
        "artifacts/cnn",
    ),
    (
        str(
            ROOT
            / "reports"
            / "final_evaluation"
            / "pre_test_freeze.json"
        ),
        "reports/final_evaluation",
    ),
]
gradio_datas = collect_data_files("gradio", include_py_files=True)
datas += [
    item
    for item in gradio_datas
    if "/gradio/test_data/"
    not in item[0].replace("\\", "/").lower()
]
datas += collect_data_files("gradio_client")
datas += collect_data_files("groovy")
datas += collect_data_files("safehttpx")

for distribution in (
    "gradio",
    "gradio-client",
    "fastapi",
    "groovy",
    "starlette",
    "pydantic",
    "safehttpx",
    "uvicorn",
):
    try:
        datas += copy_metadata(distribution)
    except Exception:
        pass

hiddenimports = [
    module
    for module in collect_submodules("gradio")
    if not module.startswith("gradio.test_data")
]
hiddenimports += collect_submodules("gradio_client")

a = Analysis(
    [str(ROOT / "windows_launcher.py")],
    pathex=[str(ROOT), str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "joblib",
        "mlflow",
        "skimage",
        "sklearn",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FontSense",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="FontSense",
)
