# Copilot instructions for PythIon

## What this repo is
- PythIon is a PyQt5 + pyqtgraph desktop GUI for nanopore/ion-channel current trace visualization and event + CUSUM “subevent state” detection.
- Most features are wired through the main window by mutating a shared app state object (`BaseAppMainWindow.perfiledata: FileData`).

## Big-picture architecture (follow these files)
- Entry points:
  - `PythIon.entrypoint:entry` runs module `PythIon.Pythion` (see `pyproject.toml`).
  - `PythIon/Pythion.py` defines `ExtAppMainWindow` and `start()`.
- GUI base + state:
  - `PythIon/BaseApp.py` builds the UI (`Ui_PythIon` from `PythIon/ui/maingui.py`) and initializes plots + shared state.
  - Shared state lives in `PythIon/DataTypes.py`:
    - `TraceData`: segmented raw/filt arrays (`_raw`, `_filt`) with `srange` segment ranges.
    - `FileData`: per-file session state (trace, baseline, samplerate, selection regions, `analysis_results`, xml records).
- Feature modules are “app-in, mutate state, paint UI”:
  - `PythIon/IO.py` load/save/export; `loadFile(app)` sets `app.perfiledata` and may parse an auxiliary `.xml`.
  - `PythIon/Analysis.py` event detection + optional CUSUM state detection (`computeAnalysis(app)`), results go into `AnalysisResults.tables`.
  - `PythIon/Painting.py` draws traces and overlays (`paintCurrentTrace(app)`, `plotAnalysis(app)`).
  - `PythIon/Selections.py` and `PythIon/Edits.py` manipulate selections/events (called from menu actions).

## Data flow you should preserve when changing behavior
- UI actions in `ExtAppMainWindow.__init__` (in `PythIon/Pythion.py`) call into modules (IO/Analysis/Painting/Selections/Edits).
- Typical analysis sequence (see `ExtAppMainWindow.doAnalysis`):
  1) `Analysis.computeAnalysis(app)` populates `app.perfiledata.analysis_results`.
  2) `Painting.plotAnalysis(app)` redraws trace + scatter/hists from result tables.
  3) Optional `IO.saveAnalysis(app)` depending on `app.enable_save_analysis`.

## File formats & external integration points
- `IO.loadFile` supports:
  - `.opt`: big-endian float64 (`dtype('>d')`)
  - `.bin`: little-endian float64 (`dtype('<d')`)
  - `.tracedata`: pickled `TraceData`
- Optional XML sidecar: `*.xml` is parsed for voltage records (`t_V_record`) and user notes.
- Filtering: Bessel lowpass via `scipy.signal` (see `IO.py`).
- CUSUM: multiprocessing + shared memory (`multiprocessing.shared_memory`) in `Analysis.py`; avoid copying large arrays.

## Project conventions (non-obvious)
- Most functions accept `app: BaseAppMainWindow` and directly update UI widgets and `app.perfiledata`; don’t refactor to “pure functions” unless you thread state through carefully.
- Plot scaling: trace Y is displayed with `current_display_scale_factor` (see `BaseApp.py` and `Painting.py`).
- UI is designed in Qt Designer: `PythIon/ui/*.ui` is the source of truth.
  - The corresponding `PythIon/ui/*.py` files are generated; avoid hand-editing them.
  - Regenerate a UI module with e.g. `pyuic5 PythIon/ui/maingui.ui -o PythIon/ui/maingui.py` (adjust filenames as needed).

## Dev workflows (what to run)
- Python version: `>=3.11, <3.12` (see `pyproject.toml`).
- Use `uv` in this repo:
  - Install deps: `uv sync`
  - Run (script entrypoint): `uv run PythIon`
  - Run (module): `uv run python -m PythIon`
- Lint (dev dependency): `ruff check .`
