import argparse
import os
import platform
from pathlib import Path
from typing import cast

import PyInstaller.__main__


def platform_tag() -> str:
    uname = platform.uname()
    os_name = uname.system.lower()
    machine = uname.machine.lower()
    return f"{os_name}-{machine}"


def build(outdir: Path, name: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    entry_script = repo_root / "scripts" / "pyinstaller_entry.py"

    if not entry_script.exists():
        raise FileNotFoundError(f"Missing entry script: {entry_script}")

    outdir.mkdir(parents=True, exist_ok=True)

    workpath = outdir / "_pyinstaller_work"
    distpath = outdir / "_pyinstaller_dist"
    specpath = outdir / "_pyinstaller_spec"

    for p in (workpath, distpath, specpath):
        p.mkdir(parents=True, exist_ok=True)

    cmd = [
        "--noconfirm",
        "--clean",
        "--onedir",
        ["--name", name],
        ["--workpath", str(workpath)],
        ["--distpath", str(distpath)],
        ["--specpath", str(specpath)],
        ["--collect-submodules", "PythIon"],
        ["--hidden-import", "PythIon.Pythion"],
        ["--collect-all", "pyqtgraph"],
        ["--collect-all", "PyQt5"],
        str(entry_script),
    ]
    cmd = cast(list[str], list(sum(cmd, [])))

    print(f"Running PyInstaller with command: {' '.join(map(str, cmd))}")
    PyInstaller.__main__.run(cmd)

    artifact_name = f"{name}-{platform_tag()}"
    artifact_path = outdir / artifact_name
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
    outdir = Path(args.out).resolve()
    artifact_path = build(outdir=outdir, name=args.name)

    print(f"Built artifact: {artifact_path}")


if __name__ == "__main__":
    os.environ.setdefault("QT_DEBUG_PLUGINS", "0")
    main()
