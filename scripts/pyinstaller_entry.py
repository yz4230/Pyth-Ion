"""PyInstaller entrypoint for PythIon.

Keep this file tiny and dependency-free so PyInstaller analysis is stable.
"""

from __future__ import annotations
from multiprocessing import freeze_support
from PythIon.Pythion import start


if __name__ == "__main__":
    freeze_support()
    start()
