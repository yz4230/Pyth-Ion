"""Compatibility wrappers that run the refactored ``calc`` engine while
producing the legacy data structures consumed by ``Painting``, ``IO``, and
the rest of the GUI.

Functions here are drop-in replacements for the corresponding legacy calls:

* ``computeAnalysis(app)``  → replaces ``Analysis.computeAnalysis``
* ``autoFindCutLRs(app)``   → replaces ``Selections.autoFindCutLRs``
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from tqdm import tqdm

from ..Analysis import AnalysisResults, Config
from ..BaseApp import BaseAppMainWindow
from .analysis import AnalysisConfig, AnalysisTables, InputConfig, analyze_tables
from .auto_detect_clears import detect_clear_regions


def _build_configs(app: BaseAppMainWindow) -> tuple[InputConfig, AnalysisConfig]:
    cfg: Config = app.analysis_config
    iconf = InputConfig(adc_samplerate_hz=app.perfiledata.ADC_samplerate_Hz)
    aconf = AnalysisConfig(
        baseline_a=cfg.baseline_A,
        baseline_std_a=cfg.baseline_std_A,
        threshold_a=cfg.threshold_A,
        enable_subevent_state_detection=cfg.enable_subevent_state_detection,
        max_states=int(cfg.maxNsta),
        cusum_stepsize=int(cfg.cusum_stepsize),
        cusum_threshhold=int(cfg.cusum_threshhold),
        merge_delta_blockade=cfg.merge_delta_blockade,
        prefilt_window_us=int(cfg.prefilt_window_us),
        state_min_duration_us=int(cfg.state_min_duration_us),
    )
    return iconf, aconf


def _populate_event_table(
    seg_result: np.ndarray,
    tables: AnalysisTables,
    *,
    id_offset: int,
    event_offset: int,
    seg_index: int,
    seg_range_start: int,
) -> None:
    ev = tables.events
    seg_result["id"] = np.arange(len(ev)) + id_offset
    seg_result["N_child"] = tables.n_children
    seg_result["category"] = "Event"
    seg_result["index"] = ev["index"].to_numpy() + event_offset
    seg_result["seg"] = seg_index
    seg_result["local_startpt"] = ev["start_point"].to_numpy()
    seg_result["local_endpt"] = ev["end_point"].to_numpy()
    seg_result["global_startpt"] = ev["start_point"].to_numpy() + seg_range_start
    seg_result["global_endpt"] = ev["end_point"].to_numpy() + seg_range_start
    seg_result["deli"] = ev["delli"].to_numpy()
    seg_result["frac"] = ev["frac"].to_numpy()
    seg_result["dwell"] = ev["dwell"].to_numpy()
    seg_result["dt"] = ev["dt"].to_numpy()
    seg_result["mean"] = ev["mean"].to_numpy()
    seg_result["stdev"] = ev["stdev"].to_numpy()
    seg_result["skewness"] = ev["skewness"].to_numpy()
    seg_result["kurtosis"] = ev["kurtosis"].to_numpy()
    seg_result["offset_first_min"] = ev["offset_to_first_min"].to_numpy()
    seg_result["stdev_tt"] = ev["stdev_tt"].to_numpy()
    seg_result["skewness_tt"] = ev["skewness_tt"].to_numpy()
    seg_result["kurtosis_tt"] = ev["kurtosis_tt"].to_numpy()


def _populate_state_table(
    state_result: np.ndarray,
    tables: AnalysisTables,
    *,
    id_offset: int,
    seg_index: int,
    seg_range_start: int,
) -> None:
    st = tables.states
    state_result["parent_id"] = st["parent_index"].to_numpy() + id_offset
    state_result["category"] = "CUSUMState"
    state_result["index"] = st["index"].to_numpy()
    state_result["seg"] = seg_index
    state_result["local_startpt"] = st["start_point"].to_numpy()
    state_result["local_endpt"] = st["end_point"].to_numpy()
    state_result["global_startpt"] = st["start_point"].to_numpy() + seg_range_start
    state_result["global_endpt"] = st["end_point"].to_numpy() + seg_range_start
    state_result["deli"] = st["delli"].to_numpy()
    state_result["frac"] = st["frac"].to_numpy()
    state_result["dwell"] = st["dwell"].to_numpy()
    state_result["mean"] = st["mean"].to_numpy()
    state_result["stdev"] = st["stdev"].to_numpy()
    state_result["skewness"] = st["skewness"].to_numpy()
    state_result["kurtosis"] = st["kurtosis"].to_numpy()


# ── public entry points ────────────────────────────────────────────


def computeAnalysis(app: BaseAppMainWindow) -> None:
    """Drop-in replacement for ``Analysis.computeAnalysis`` using the
    refactored ``calc.analysis`` engine."""
    app.printlog("Computing Analysis (refactored engine)...")
    analysis_config: Config = app.analysis_config
    analysis_results = AnalysisResults(analysis_config)
    app.printlog(f"Subevent analysis with configuration: \n{analysis_config}\n")

    result_nullvalue = analysis_results.result_nullvalue
    result_dtype = analysis_results.result_dtype
    event_result_table = analysis_results.newResultTable()
    state_result_table = analysis_results.newResultTable()

    iconf, aconf = _build_configs(app)

    id_counter = 0
    event_index = 0

    for k in tqdm(range(app.perfiledata.data.Nseg), desc="Segment->Event(refactored)"):
        seg_filt = app.perfiledata.data.filt[k]
        seg_range = app.perfiledata.data.srange[k]

        tables = analyze_tables(data=seg_filt, iconf=iconf, aconf=aconf)
        seg_n = len(tables.events)

        if seg_n > 0:
            seg_result = np.full(seg_n, result_nullvalue, dtype=result_dtype)
            _populate_event_table(
                seg_result,
                tables,
                id_offset=id_counter,
                event_offset=event_index,
                seg_index=k,
                seg_range_start=int(seg_range[0]),
            )
            event_result_table = np.append(event_result_table, seg_result)

        if (
            analysis_config.enable_subevent_state_detection
            and not tables.states.empty
        ):
            state_n = len(tables.states)
            state_result = np.full(state_n, result_nullvalue, dtype=result_dtype)
            _populate_state_table(
                state_result,
                tables,
                id_offset=id_counter,
                seg_index=k,
                seg_range_start=int(seg_range[0]),
            )
            state_result_table = np.append(state_result_table, state_result)

        event_index += seg_n
        id_counter += seg_n

    if analysis_config.enable_subevent_state_detection:
        state_result_table["id"] = np.arange(len(state_result_table)) + id_counter
        id_counter += len(state_result_table)
        analysis_results.tables["CUSUMState"] = state_result_table

    analysis_results.tables["Event"] = event_result_table
    app.perfiledata.analysis_results = analysis_results
    app.printlog(
        f"Analyzed with baseline: {analysis_config.baseline_A * 1e9:.4f} nA, "
        f"std: {analysis_config.baseline_std_A * 1e9:.5f} nA, "
        f"threshold: {analysis_config.threshold_A * 1e9:.4f} nA.\n"
        f"Total events found: {len(analysis_results.tables['Event']):d}."
    )


def autoFindCutLRs(app: BaseAppMainWindow) -> None:
    """Drop-in replacement for ``Selections.autoFindCutLRs`` using the
    refactored ``calc.auto_detect_clears`` engine."""
    with app.awaitresponse:
        for k in range(app.perfiledata.data.Nseg):
            seg_range = app.perfiledata.data.srange[k]
            seg_filt = app.perfiledata.data.filt[k]

            regions = detect_clear_regions(
                seg_filt,
                baseline=app.ui_baseline,
                baseline_std=app.ui_baseline_std,
                spike_std_threshold=10.0,
                baseline_window_std=1.0,
                sample_rate_hz=app.perfiledata.ADC_samplerate_Hz,
                extra_relaxation_ms=10.0,
            )

            for region_start, region_end in regions:
                newLR = pg.LinearRegionItem()
                newLR.hide()
                # calc returns half-open [start, end); legacy uses inclusive end
                local_region = np.array((region_start, region_end - 1))
                global_region = local_region + seg_range[0]
                newLR.setRegion(global_region)
                app.p1.addItem(newLR)
                app.perfiledata.LRs.append(newLR)
                newLR.show()
