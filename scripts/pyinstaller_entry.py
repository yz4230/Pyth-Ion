"""PyInstaller entrypoint for PythIon.

Keep this file tiny and dependency-free so PyInstaller analysis is stable.
"""

from __future__ import annotations
import multiprocessing as mp


def main():
    from PythIon.Pythion import start

    start()


if __name__ == "__main__":
    mp.freeze_support()
    main()
