from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _platform_tag() -> str:
    # Align with common CI naming: windows / darwin / linux
    if sys.platform.startswith("win"):
        os_name = "windows"
    elif sys.platform == "darwin":
        os_name = "darwin"
    else:
        os_name = "linux"

    machine = platform.machine().lower()
    # Normalize common variants
    machine = {
        "amd64": "amd64",
        "x86_64": "amd64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }.get(machine, machine)

    return f"{os_name}-{machine}"


def _exe_suffix() -> str:
    return ".exe" if sys.platform.startswith("win") else ""


def build_onedir(
    out_dir: Path,
    name: str = "PythIon",
) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    entry_script = repo_root / "scripts" / "pyinstaller_entry.py"

    if not entry_script.exists():
        raise FileNotFoundError(f"Missing entry script: {entry_script}")

    out_dir.mkdir(parents=True, exist_ok=True)

    workpath = out_dir / "_pyinstaller_work"
    distpath = out_dir / "_pyinstaller_dist"
    specpath = out_dir / "_pyinstaller_spec"

    for p in (workpath, distpath, specpath):
        p.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        name,
        "--workpath",
        str(workpath),
        "--distpath",
        str(distpath),
        "--specpath",
        str(specpath),
        # TODO: Some features may require a console
        # "--windowed",
        # entrypoint.py uses runpy.run_module("PythIon.Pythion", ...), which PyInstaller
        # cannot statically analyze. Force inclusion of app submodules.
        "--collect-submodules",
        "PythIon",
        "--hidden-import",
        "PythIon.Pythion",
        # Be conservative: PyQt5 + pyqtgraph sometimes need extra collection.
        "--collect-all",
        "pyqtgraph",
        "--collect-all",
        "PyQt5",
        str(entry_script),
    ]

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(repo_root), check=True)

    platform_tag = _platform_tag()

    # macOS can produce either:
    # - an .app bundle (typically with --windowed)
    # - a normal onedir folder (console mode)
    if sys.platform == "darwin":
        app_bundle = distpath / f"{name}.app"
        onedir_dir = distpath / name

        if app_bundle.exists():
            built_dir = app_bundle
            artifact_path = out_dir / f"{name}-{platform_tag}.app"
        elif onedir_dir.exists():
            built_dir = onedir_dir
            artifact_path = out_dir / f"{name}-{platform_tag}"
        else:
            raise FileNotFoundError(
                "Expected PyInstaller output not found. Looked for: "
                f"{app_bundle} and {onedir_dir}"
            )
    else:
        built_dir = distpath / name
        if not built_dir.exists():
            raise FileNotFoundError(
                f"Expected PyInstaller onedir output not found: {built_dir}"
            )
        artifact_path = out_dir / f"{name}-{platform_tag}"

    if artifact_path.exists():
        if artifact_path.is_dir():
            shutil.rmtree(artifact_path)
        else:
            artifact_path.unlink()

    shutil.move(str(built_dir), str(artifact_path))

    # Keep artifacts dir clean: remove intermediate build folders.
    for p in (workpath, specpath, distpath):
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)

    # Sanity check: make sure the executable exists within onedir output
    if sys.platform.startswith("win"):
        expected_exe = artifact_path / f"{name}{_exe_suffix()}"
        if not expected_exe.exists():
            raise FileNotFoundError(
                f"Expected executable not found in onedir output: {expected_exe}"
            )

    return artifact_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="build",
        help="Output directory for CI artifacts (default: build)",
    )
    parser.add_argument(
        "--name",
        default="PythIon",
        help="Base name for the produced binary (default: PythIon)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    artifact_path = build_onedir(
        out_dir=out_dir,
        name=args.name,
    )

    print(f"Built artifact: {artifact_path}")


if __name__ == "__main__":
    # Avoid Qt platform plugin errors due to missing DISPLAY on some CI hosts.
    # (Harmless on Windows/macOS; ignored if not applicable.)
    os.environ.setdefault("QT_DEBUG_PLUGINS", "0")
    main()
