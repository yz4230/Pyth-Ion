# Copilot instructions for PythIon

## What this repo is

- PythIon is a PyQt5 + pyqtgraph desktop GUI for nanopore/ion-channel current trace visualization and event + CUSUM "subevent state" detection.
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
- Feature modules are "app-in, mutate state, paint UI":
  - `PythIon/IO.py` load/save/export; `loadFile(app)` sets `app.perfiledata` and may parse an auxiliary `.xml`.
  - `PythIon/Analysis.py` event detection + optional CUSUM state detection (`computeAnalysis(app)`), results go into `AnalysisResults.tables`.
  - `PythIon/Painting.py` draws traces and overlays (`paintCurrentTrace(app)`, `plotAnalysis(app)`).
  - `PythIon/Selections.py` and `PythIon/Edits.py` manipulate selections/events (called from menu actions).
- Refactored analysis engine (alternative implementation):
  - `PythIon/calc/analysis.py` contains `AnalysisTables` dataclass and `analyze_tables()` function.
  - `PythIon/calc/compat.py` provides compatibility layer to convert between refactored and legacy data structures.
  - Enable via menu: Analysis > Use Refactored Engine.

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

- Most functions accept `app: BaseAppMainWindow` and directly update UI widgets and `app.perfiledata`; don't refactor to "pure functions" unless you thread state through carefully.
- Plot scaling: trace Y is displayed with `current_display_scale_factor` (see `BaseApp.py` and `Painting.py`).
- UI is designed in Qt Designer: `PythIon/ui/*.ui` is the source of truth.
  - The corresponding `PythIon/ui/*.py` files are generated; avoid hand-editing them.
  - Regenerate a UI module with e.g. `pyuic5 PythIon/ui/maingui.ui -o PythIon/ui/maingui.py` (adjust filenames as needed).

## Analysis result structure

### Legacy implementation (`Analysis.py`)

- `AnalysisResults` class with `result_spec` defining structured array fields:
  - Event fields: `id`, `N_child`, `parent_id`, `category`, `index`, `seg`, `local_startpt`, `local_endpt`, `global_startpt`, `global_endpt`, `deli`, `frac`, `dwell`, `dt`, `mean`, `stdev`, `skewness`, `kurtosis`, `offset_first_min`, `stdev_tt`, `skewness_tt`, `kurtosis_tt`, `fft_mean`
- `AnalysisResults.tables` is a `dict[str, np.ndarray]`:
  - `"Event"`: parent event records (category="Event")
  - `"CUSUMState"`: subevent state records (category="CUSUMState")

### Refactored implementation (`calc/analysis.py`)

- `AnalysisTables` dataclass with:
  - `events`: pandas DataFrame with columns defined in `EVENT_HEADERS`
  - `states`: pandas DataFrame with columns defined in `STATE_HEADERS`
  - `n_children`: numpy array of child counts per event
- `calc/compat.py` converts `AnalysisTables` to legacy format via `_populate_event_table()` and `_populate_state_table()`.

## Scatter plot structure

- Scatter plots are organized as dicts with entries: `"events"`, `"cusum_states"`, `"annotations"`
- Plot widgets defined in `BaseApp.py`:
  - `p2` (frac), `p2std` (stdev), `p2skew` (skewness), `p2kurt` (kurtosis), `p2fft` (FFT mean)
- All scatter plot dicts are collected in `self.p2s` tuple for batch operations (clearing, click handling).
- Click handling: `sigClicked` signals connected in `Pythion.py` → `Painting.scatterClicked()` → `inspectEvent()`.

## Adding new features

- For new event features: see `docs/adding_new_graph.md`
- Key files to modify:
  1. `Analysis.py`: Add field to `result_spec`, compute in event loop
  2. `calc/analysis.py`: Add to `EVENT_HEADERS`, compute in `analyze_tables()`
  3. `calc/compat.py`: Add mapping in `_populate_event_table()`
  4. `ui/maingui.ui`: Add plot widget (use Qt Designer)
  5. `BaseApp.py`: Initialize plot widget, add to `p2s`
  6. `Painting.py`: Add scatter plot drawing and annotation handling

## Dev workflows (what to run)

- Python version: `>=3.11, <3.12` (see `pyproject.toml`).
- Use `uv` in this repo:
  - Install deps: `uv sync`
  - Run (script entrypoint): `uv run PythIon`
  - Run (module): `uv run python -m PythIon`
- Lint (dev dependency): `uv run ruff check .`
- Regenerate UI: `pyuic5 PythIon/ui/maingui.ui -o PythIon/ui/maingui.py`

## Additional documentation

The `docs/` directory contains additional documentation:

- `docs/codebase_overview.md`: High-level overview of the codebase structure
- `docs/cusum.md`: Detailed explanation of the CUSUM algorithm for subevent detection
- `docs/adding_new_graph.md`: Step-by-step guide for adding new event feature graphs
