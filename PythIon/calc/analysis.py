import numpy as np
from dataclasses import dataclass

import pandas as pd
from scipy import signal, stats


@dataclass(kw_only=True, frozen=True)
class InputConfig:
    adc_samplerate_hz: int
    lpf_cutoff_hz: int = 100


@dataclass(kw_only=True, frozen=True)
class AnalysisConfig:
    baseline_a: float = np.nan
    baseline_std_a: float = np.nan
    threshold_a: float = 0.3e-9
    enable_subevent_state_detection: bool = True
    max_states: int = 16
    cusum_stepsize: int = 10
    cusum_threshhold: int = 30
    merge_delta_blockade: float = 0.02
    prefilt_window_us: int = 100
    state_min_duration_us: int = 150


DF_HEADERS = [
    "index",
    "start_point",
    "end_point",
    "delli",
    "frac",
    "dwell",
    "dt",
    "mean",
    "stdev",
    "skewness",
    "kurtosis",
    "offset_to_first_min",
    "stdev_tt",
    "skewness_tt",
    "kurtosis_tt",
]


def analyze(*, data: np.ndarray, iconf: InputConfig, aconf: AnalysisConfig):
    if iconf.adc_samplerate_hz <= 0:
        raise ValueError("adc_samplerate_hz must be positive")

    # cusum_min_len = int(aconf.state_min_duration_us * iconf.adc_samplerate_hz * 1e-6)
    # prefilt_oneside_window = int(aconf.prefilt_window_us / 2 * iconf.adc_samplerate_hz * 1e-6)  # fmt: skip
    # cusum_max_states = aconf.max_states - 1
    # merge_delta_i = aconf.merge_delta_blockade * aconf.baseline_A

    (below,) = np.where(data < aconf.threshold_a)
    start_and_end = np.diff(below)
    if len(start_and_end) == 0:
        return pd.DataFrame(columns=DF_HEADERS)

    start_points = np.insert(start_and_end, 0, 2)
    end_points = np.insert(start_and_end, -1, 2)
    (start_points,) = np.where(start_points > 1)
    (end_points,) = np.where(end_points > 1)
    start_points = below[start_points]
    end_points = below[end_points]

    if start_points.size == 0:
        return pd.DataFrame(columns=DF_HEADERS)
    if start_points[0] == 0:
        start_points = start_points[1:]
        end_points = end_points[1:]
    if end_points.size == 0:
        return pd.DataFrame(columns=DF_HEADERS)
    if end_points[-1] == data.size - 1:
        start_points = start_points[:-1]
        end_points = end_points[:-1]

    number_of_events = start_points.size
    high_thresh = aconf.baseline_a - aconf.baseline_std_a

    for i in range(number_of_events):
        sp = start_points[i]
        while sp > 0 and data[sp] < high_thresh:
            sp -= 1
        start_points[i] = sp

        ep = end_points[i]
        if ep == data.size - 1:
            start_points[i] = 0
            end_points[i] = 0
            ep = 0
            break

        while data[ep] < high_thresh:
            ep += 1
            if ep == data.size - 1:
                start_points[i] = 0
                end_points[i] = 0
                ep = 0
                break

            if i + 1 < number_of_events and ep > start_points[i + 1]:
                start_points[i + 1] = 0
                end_points[i] = 0
                ep = 0
                break

        end_points[i] = ep

    start_points = start_points[start_points != 0]
    end_points = end_points[end_points != 0]
    number_of_events = start_points.size

    if start_points.size > end_points.size:
        start_points = start_points[: end_points.size]
        number_of_events = start_points.size

    delis = np.zeros(number_of_events)
    dwells = np.zeros(number_of_events)
    first_min_offsets = np.full(number_of_events, -1, dtype=np.int32)

    for i in range(number_of_events):
        (relmin,) = signal.argrelmin(data[start_points[i] : end_points[i]])
        mins = np.array(relmin + start_points[i])
        cut = (aconf.baseline_a + np.mean(data[start_points[i] : end_points[i]])) / 2
        mins = mins[data[mins] < cut]
        if len(mins) == 1:
            delis[i] = aconf.baseline_a - min(data[start_points[i] : end_points[i]])
            dwells[i] = ((end_points[i] - start_points[i]) / iconf.adc_samplerate_hz * 1e6)  # fmt: skip
            end_points[i] = mins[0]
            first_min_offsets[i] = -2
        elif len(mins) > 1:
            delis[i] = aconf.baseline_a - np.mean(data[mins[0] : mins[-1]])
            end_points[i] = mins[-1]
            dwells[i] = (end_points[i] - start_points[i]) / iconf.adc_samplerate_hz * 1e6  # fmt: skip
            first_min_offsets[i] = mins[0] - start_points[i]

    valid_events = np.logical_and(delis != 0, dwells != 0)
    start_points = start_points[valid_events]
    end_points = end_points[valid_events]
    first_min_offsets = first_min_offsets[valid_events]
    delis = delis[valid_events]
    dwells = dwells[valid_events]
    fracs = delis / aconf.baseline_a
    number_of_events = start_points.size
    if number_of_events == 0:
        return pd.DataFrame(columns=DF_HEADERS)

    dts = np.empty(number_of_events, dtype=np.float64)
    dts[0] = np.nan
    if number_of_events > 1:
        dts[1:] = np.diff(start_points) / iconf.adc_samplerate_hz

    baseline = np.empty(number_of_events, dtype=np.float64)
    noise = np.empty(number_of_events, dtype=np.float64)
    skew = np.empty(number_of_events, dtype=np.float64)
    kurt = np.empty(number_of_events, dtype=np.float64)
    stdev_tt = np.full(number_of_events, np.nan)
    skew_tt = np.full(number_of_events, np.nan)
    kurt_tt = np.full(number_of_events, np.nan)

    for idx, (start_point, end_point) in enumerate(zip(start_points, end_points)):
        segment = data[start_point:end_point]
        baseline[idx] = np.mean(segment)
        noise[idx] = np.std(segment)
        skew[idx] = stats.skew(segment)
        kurt[idx] = stats.kurtosis(segment)

        first_min_offset = first_min_offsets[idx]
        if first_min_offset > 0:
            trough = data[start_point + first_min_offset : end_point]
            if trough.size != 0:
                stdev_tt[idx] = np.std(trough)
                skew_tt[idx] = stats.skew(trough)
                kurt_tt[idx] = stats.kurtosis(trough)

    return pd.DataFrame(
        {
            "index": np.arange(number_of_events),
            "start_point": start_points,
            "end_point": end_points,
            "delli": delis,
            "dwell": dwells,
            "frac": fracs,
            "dt": dts,
            "mean": baseline,
            "stdev": noise,
            "skewness": skew,
            "kurtosis": kurt,
            "offset_to_first_min": first_min_offsets,
            "stdev_tt": stdev_tt,
            "skewness_tt": skew_tt,
            "kurtosis_tt": kurt_tt,
        }
    )
