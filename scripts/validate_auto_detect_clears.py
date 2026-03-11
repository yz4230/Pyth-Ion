from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import signal

from PythIon.calc.auto_detect_clears import detect_clear_regions


DEFAULT_DATA_PATH = Path("~/nanopore/data/B122625_004.opt").expanduser()
DEFAULT_ADC_SAMPLERATE_HZ = 250_000
DEFAULT_LPF_CUTOFF_HZ = 100_000


def load_opt_trace(path: Path, adc_samplerate_hz: int, lpf_cutoff_hz: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.dtype(">d"))
    wn = round(lpf_cutoff_hz / (adc_samplerate_hz / 2), 4)
    b, a = signal.bessel(4, wn, btype="low")
    return signal.filtfilt(b, a, raw)


def run_legacy_auto_detect(
    filt: np.ndarray,
    baseline: float,
    baseline_std: float,
    adc_samplerate_hz: int,
) -> np.ndarray:
    """Replicate Selections.autoFindCutLRs logic without GUI dependency."""
    std_thresh_for_spike = 10
    ndx_spike = np.where(
        np.abs(filt) > baseline + std_thresh_for_spike * baseline_std
    )[0]
    ndx_around_baseline = np.nonzero(
        np.abs(filt - baseline) < baseline_std
    )[0]
    extra_relaxation_time_ms = 10
    Nrlx = int(extra_relaxation_time_ms * adc_samplerate_hz / 1e3)

    legacy_regions: list[tuple[int, int]] = []
    k_nab = 0
    endpoint = 0
    for startpoint in ndx_spike:
        if startpoint < endpoint:
            continue
        k_nab += int(
            np.searchsorted(ndx_around_baseline[k_nab:], startpoint, side="right")
        )
        endpoint = int(ndx_around_baseline[k_nab + Nrlx])
        if k_nab > 0:
            startpoint = int(ndx_around_baseline[k_nab - 1])
        else:
            startpoint = 0
        legacy_regions.append((startpoint, endpoint))

    if not legacy_regions:
        return np.empty((0, 2), dtype=int)
    return np.array(legacy_regions, dtype=int)


def run_validation(path: Path, adc_samplerate_hz: int, lpf_cutoff_hz: int) -> None:
    filt = load_opt_trace(path, adc_samplerate_hz=adc_samplerate_hz, lpf_cutoff_hz=lpf_cutoff_hz)
    baseline = float(np.median(filt))
    baseline_std = float(np.std(filt))

    legacy_regions = run_legacy_auto_detect(
        filt,
        baseline=baseline,
        baseline_std=baseline_std,
        adc_samplerate_hz=adc_samplerate_hz,
    )

    calc_regions = detect_clear_regions(
        filt,
        baseline=baseline,
        baseline_std=baseline_std,
        spike_std_threshold=10.0,
        baseline_window_std=1.0,
        sample_rate_hz=adc_samplerate_hz,
        extra_relaxation_ms=10.0,
    )

    if len(legacy_regions) != len(calc_regions):
        raise AssertionError(
            f"region count mismatch: legacy={len(legacy_regions)} calc={len(calc_regions)}"
        )

    if not np.array_equal(legacy_regions[:, 0], calc_regions[:, 0]):
        raise AssertionError("start point mismatch")

    # calc uses half-open [start, end); legacy uses inclusive end
    if not np.array_equal(legacy_regions[:, 1] + 1, calc_regions[:, 1]):
        raise AssertionError("end point mismatch (expected calc_end == legacy_end + 1)")

    print(f"data_path={path}")
    print(f"samples={len(filt)}")
    print(f"baseline={baseline:.18e}")
    print(f"baseline_std={baseline_std:.18e}")
    print(f"region_count={len(calc_regions)}")
    print("auto-detect-clears parity ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--adc-samplerate-hz", type=int, default=DEFAULT_ADC_SAMPLERATE_HZ)
    parser.add_argument("--lpf-cutoff-hz", type=int, default=DEFAULT_LPF_CUTOFF_HZ)
    args = parser.parse_args()

    run_validation(
        path=args.data_path.expanduser(),
        adc_samplerate_hz=args.adc_samplerate_hz,
        lpf_cutoff_hz=args.lpf_cutoff_hz,
    )


if __name__ == "__main__":
    main()
