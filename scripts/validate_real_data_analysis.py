from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import signal

from PythIon.Analysis import Config, computeAnalysis
from PythIon.calc.analysis import AnalysisConfig, InputConfig, analyze_tables


DEFAULT_DATA_PATH = Path("~/nanopore/data/B122625_004.opt").expanduser()
DEFAULT_ADC_SAMPLERATE_HZ = 250_000
DEFAULT_LPF_CUTOFF_HZ = 100_000


def load_opt_trace(path: Path, adc_samplerate_hz: int, lpf_cutoff_hz: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.dtype(">d"))
    wn = round(lpf_cutoff_hz / (adc_samplerate_hz / 2), 4)
    b, a = signal.bessel(4, wn, btype="low")
    return signal.filtfilt(b, a, raw)


def build_legacy_app(
    data: np.ndarray,
    adc_samplerate_hz: int,
    baseline_a: float,
    baseline_std_a: float,
    threshold_a: float,
) -> SimpleNamespace:
    config = Config()
    config.baseline_A = baseline_a
    config.baseline_std_A = baseline_std_a
    config.threshold_A = threshold_a

    perfiledata = SimpleNamespace(
        ADC_samplerate_Hz=adc_samplerate_hz,
        data=SimpleNamespace(
            Nseg=1,
            filt=[data],
            srange=[np.array([0, len(data)], dtype=np.int64)],
        ),
    )
    return SimpleNamespace(
        analysis_config=config,
        perfiledata=perfiledata,
        printlog=lambda *args, **kwargs: None,
    )


def assert_frame_matches(
    calc_df: pd.DataFrame, legacy_table: np.ndarray, mapping: dict[str, str]
) -> None:
    if len(calc_df) != len(legacy_table):
        raise AssertionError(
            f"row count mismatch: calc={len(calc_df)} legacy={len(legacy_table)}"
        )

    for calc_col, legacy_col in mapping.items():
        calc = calc_df[calc_col].to_numpy()
        legacy = legacy_table[legacy_col]
        if np.issubdtype(calc.dtype, np.floating):
            if not np.allclose(calc, legacy, equal_nan=True, atol=1e-12, rtol=1e-9):
                raise AssertionError(f"column mismatch: {calc_col}")
        elif not np.array_equal(calc, legacy):
            raise AssertionError(f"column mismatch: {calc_col}")


def run_validation(
    path: Path, adc_samplerate_hz: int, lpf_cutoff_hz: int, threshold_a: float
) -> None:
    filt = load_opt_trace(path, adc_samplerate_hz=adc_samplerate_hz, lpf_cutoff_hz=lpf_cutoff_hz)
    baseline_a = float(np.median(filt))
    baseline_std_a = float(np.std(filt))

    app = build_legacy_app(
        filt,
        adc_samplerate_hz=adc_samplerate_hz,
        baseline_a=baseline_a,
        baseline_std_a=baseline_std_a,
        threshold_a=threshold_a,
    )
    computeAnalysis(app)
    legacy_results = app.perfiledata.analysis_results

    calc_results = analyze_tables(
        data=filt,
        iconf=InputConfig(adc_samplerate_hz=adc_samplerate_hz),
        aconf=AnalysisConfig(
            baseline_a=baseline_a,
            baseline_std_a=baseline_std_a,
            threshold_a=threshold_a,
        ),
    )

    event_mapping = {
        "index": "index",
        "start_point": "local_startpt",
        "end_point": "local_endpt",
        "delli": "deli",
        "frac": "frac",
        "dwell": "dwell",
        "dt": "dt",
        "mean": "mean",
        "stdev": "stdev",
        "skewness": "skewness",
        "kurtosis": "kurtosis",
        "offset_to_first_min": "offset_first_min",
        "stdev_tt": "stdev_tt",
        "skewness_tt": "skewness_tt",
        "kurtosis_tt": "kurtosis_tt",
    }
    state_mapping = {
        "parent_index": "parent_id",
        "index": "index",
        "start_point": "local_startpt",
        "end_point": "local_endpt",
        "delli": "deli",
        "frac": "frac",
        "dwell": "dwell",
        "mean": "mean",
        "stdev": "stdev",
        "skewness": "skewness",
        "kurtosis": "kurtosis",
    }

    legacy_events = legacy_results.tables["Event"]
    assert_frame_matches(calc_results.events, legacy_events, event_mapping)
    if not np.array_equal(calc_results.n_children, legacy_events["N_child"]):
        raise AssertionError("N_child mismatch")

    legacy_states = legacy_results.tables.get("CUSUMState")
    if legacy_states is None:
        if not calc_results.states.empty:
            raise AssertionError("calc produced CUSUMState rows, legacy did not")
    else:
        assert_frame_matches(calc_results.states, legacy_states, state_mapping)

    print(f"data_path={path}")
    print(f"samples={len(filt)}")
    print(f"baseline_a={baseline_a:.18e}")
    print(f"baseline_std_a={baseline_std_a:.18e}")
    print(f"threshold_a={threshold_a:.18e}")
    print(f"event_count={len(calc_results.events)}")
    print(f"state_count={len(calc_results.states)}")
    print("real-data parity ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--adc-samplerate-hz", type=int, default=DEFAULT_ADC_SAMPLERATE_HZ)
    parser.add_argument("--lpf-cutoff-hz", type=int, default=DEFAULT_LPF_CUTOFF_HZ)
    parser.add_argument("--threshold-a", type=float, default=3e-10)
    args = parser.parse_args()

    run_validation(
        path=args.data_path.expanduser(),
        adc_samplerate_hz=args.adc_samplerate_hz,
        lpf_cutoff_hz=args.lpf_cutoff_hz,
        threshold_a=args.threshold_a,
    )


if __name__ == "__main__":
    main()
