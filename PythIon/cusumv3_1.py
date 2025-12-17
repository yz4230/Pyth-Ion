"""
CUSUM (Cumulative Sum) Algorithm for Change Point Detection

This module implements a CUSUM-based algorithm for detecting abrupt changes
in time series data, commonly used in nanopore sensing and ion channel analysis.
"""

from dataclasses import dataclass
from typing import Optional, TypedDict

import numpy as np
import numpy.typing as npt
from scipy.ndimage import median_filter as spmedfilt


class CUSUMResultDict(TypedDict):
    """Type definition for CUSUM detection result dictionary."""

    nStates: int
    starts: npt.NDArray[np.int64]
    threshold: float
    stepsize: float


@dataclass
class CUSUMResult:
    """Result container for CUSUM detection."""

    n_states: int
    starts: npt.NDArray[np.int64]
    threshold: float
    stepsize: float

    def to_dict(self) -> CUSUMResultDict:
        """Convert to dictionary format for backward compatibility."""
        return {
            "nStates": self.n_states,
            "starts": self.starts,
            "threshold": self.threshold,
            "stepsize": self.stepsize,
        }


def _central_moving_average(
    data: npt.NDArray[np.floating],
    window: int,
) -> Optional[npt.NDArray[np.float64]]:
    """
    Compute central moving average with edge handling.

    Parameters
    ----------
    data : npt.NDArray[np.floating]
        Input data array.
    window : int
        One-sided window size.

    Returns
    -------
    npt.NDArray[np.float64] or None
        Smoothed data, or None if data is too short.
    """
    full_window = 2 * window + 1
    if len(data) <= full_window:
        return None

    # Scale data to int64 to avoid floating point accumulation errors
    int32_max = np.iinfo(np.int32).max
    data_absmax = np.abs(data).max()
    if data_absmax == 0:
        return np.zeros_like(data, dtype=np.float64)

    scaled_data = (data / data_absmax * int32_max).astype(np.int64)
    cumsum = np.cumsum(scaled_data)

    result = np.zeros_like(scaled_data)

    # Left edge: expanding window
    result[: window + 1] = cumsum[window:full_window]

    # Center: full window
    result[window + 1 : -window] = cumsum[full_window:] - cumsum[:-full_window]

    # Right edge: contracting window
    result[-window:] = cumsum[-1] - cumsum[-full_window : -window - 1]

    # Convert back to float and normalize
    result = result.astype(np.float64) / int32_max * data_absmax

    # Adjust for varying window sizes at edges
    left_counts = np.arange(window + 1, full_window + 1)
    right_counts = np.arange(full_window - 1, window, -1)

    result[: window + 1] /= left_counts
    result[window + 1 : -window] /= full_window
    result[-window:] /= right_counts

    return result


def _central_moving_median(
    data: npt.NDArray[np.floating],
    window: int,
) -> Optional[npt.NDArray[np.floating]]:
    """
    Compute central moving median with edge padding.

    Parameters
    ----------
    data : npt.NDArray[np.floating]
        Input data array.
    window : int
        One-sided window size.

    Returns
    -------
    npt.NDArray[np.floating] or None
        Filtered data, or None if data is too short.
    """
    full_window = 2 * window + 1
    if len(data) <= full_window:
        return None

    # Pad data with edge values
    padded = np.empty(len(data) + 2 * window)
    padded[:window] = data[:window]
    padded[window:-window] = data
    padded[-window:] = data[-window:]

    # Apply median filter
    filtered = spmedfilt(padded, full_window)

    return filtered[window:-window]


def _preprocess_data(
    data: npt.NDArray[np.floating],
    window: int,
) -> Optional[npt.NDArray[np.floating]]:
    """
    Apply moving average and median filtering for noise reduction.

    Parameters
    ----------
    data : npt.NDArray[np.floating]
        Raw input data.
    window : int
        One-sided window size for filtering.

    Returns
    -------
    npt.NDArray[np.floating] or None
        Preprocessed data, or None if data is too short.
    """
    if window <= 0:
        return data.copy()

    smoothed = _central_moving_average(data, window)
    if smoothed is None:
        return None

    return _central_moving_median(smoothed, window)


def _create_single_state_result(
    data_length: int,
    threshold: float,
    stepsize: float,
) -> CUSUMResultDict:
    """Create result for single-state (no change points) case."""
    return {
        "nStates": 1,
        "starts": np.array([0, data_length], dtype=np.int64),
        "threshold": threshold,
        "stepsize": stepsize,
    }


def _compute_log_likelihoods(
    current_value: float,
    mean: float,
    variance: float,
    base_sd: float,
    stepsize: float,
) -> tuple[float, float]:
    """
    Compute instantaneous log-likelihoods for positive and negative jumps.

    Parameters
    ----------
    current_value : float
        Current data point.
    mean : float
        Running mean of data since last anchor.
    variance : float
        Running variance of data since last anchor.
    base_sd : float
        Baseline standard deviation.
    stepsize : float
        Expected jump size in units of base_sd.

    Returns
    -------
    tuple[float, float]
        (log_likelihood_positive, log_likelihood_negative)
    """
    delta = stepsize * base_sd
    scale = delta / variance
    deviation = current_value - mean

    log_pos = scale * (deviation - delta / 2)
    log_neg = -scale * (deviation + delta / 2)

    return log_pos, log_neg


class WelfordVariance:
    """Online algorithm for computing running mean and variance (Welford's method)."""

    mean: float
    _m: float
    _s: float
    _count: int

    def __init__(self, initial_value: float) -> None:
        self.mean = initial_value
        self._m = initial_value
        self._s = 0.0
        self._count = 1

    def update(self, value: float) -> None:
        """Add a new value and update statistics."""
        self._count += 1
        old_m = self._m
        self._m += (value - self._m) / self._count
        self._s += (value - old_m) * (value - self._m)
        self.mean = self._m

    @property
    def variance(self) -> float:
        """Current variance estimate."""
        return self._s / self._count if self._count > 0 else 0.0

    def reset(self, initial_value: float) -> None:
        """Reset statistics with a new initial value."""
        self.__init__(initial_value)


class CUSUMDetector:
    """
    CUSUM-based change point detector.

    This detector identifies abrupt changes in time series data using
    cumulative sum statistics with adaptive thresholding.
    """

    base_sd: float
    threshold: float
    stepsize: float
    min_length: int
    _edges: list[int]
    _anchor: int
    _cpos: Optional[npt.NDArray[np.float64]]
    _cneg: Optional[npt.NDArray[np.float64]]
    _gpos: Optional[npt.NDArray[np.float64]]
    _gneg: Optional[npt.NDArray[np.float64]]
    _stats: Optional[WelfordVariance]

    def __init__(
        self,
        base_sd: float,
        threshold: float = 10.0,
        stepsize: float = 3.0,
        min_length: int = 1000,
    ) -> None:
        self.base_sd = base_sd
        self.threshold = threshold
        self.stepsize = stepsize
        self.min_length = min_length
        self._reset()

    def _reset(self) -> None:
        """Reset internal state for a new detection run."""
        self._edges = [0]
        self._anchor = 0
        self._cpos = None
        self._cneg = None
        self._gpos = None
        self._gneg = None
        self._stats = None

    def _initialize_arrays(self, length: int) -> None:
        """Initialize cumulative and decision arrays."""
        self._cpos = np.zeros(length, dtype=np.float64)
        self._cneg = np.zeros(length, dtype=np.float64)
        self._gpos = np.zeros(length, dtype=np.float64)
        self._gneg = np.zeros(length, dtype=np.float64)

    def _reset_arrays(self) -> None:
        """Reset all cumulative arrays to zero."""
        assert self._cpos is not None and self._cneg is not None
        assert self._gpos is not None and self._gneg is not None
        self._cpos.fill(0)
        self._cneg.fill(0)
        self._gpos.fill(0)
        self._gneg.fill(0)

    def _find_jump_location(
        self,
        cumsum_array: npt.NDArray[np.float64],
        end: int,
    ) -> int:
        """Find the location of jump start using minimum of cumulative sum."""
        return int(self._anchor + np.argmin(cumsum_array[self._anchor : end + 1]))

    def _process_jump(self, jump_location: int) -> bool:
        """
        Process a detected jump and add edge if valid.

        Returns True if edge was added.
        """
        n_states = len(self._edges) - 1
        if jump_location - self._edges[n_states] > self.min_length:
            self._edges.append(jump_location)
            return True
        return False

    def _detect_single_pass(self, data: npt.NDArray[np.floating]) -> int:
        """
        Run a single pass of CUSUM detection.

        Returns the number of detected states.
        """

        length = len(data)
        self._initialize_arrays(length)
        self._stats = WelfordVariance(data[0])
        base_variance = self.base_sd**2

        assert self._cpos is not None and self._cneg is not None
        assert self._gpos is not None and self._gneg is not None

        for k in range(1, length):
            self._stats.update(data[k])

            # Use baseline variance if running variance is zero
            variance = (
                self._stats.variance if self._stats.variance > 0 else base_variance
            )

            # Compute log-likelihoods
            log_pos, log_neg = _compute_log_likelihoods(
                data[k], self._stats.mean, variance, self.base_sd, self.stepsize
            )

            # Update cumulative sums
            self._cpos[k] = self._cpos[k - 1] + log_pos
            self._cneg[k] = self._cneg[k - 1] + log_neg

            # Update decision functions (reset to 0 if negative)
            self._gpos[k] = max(self._gpos[k - 1] + log_pos, 0)
            self._gneg[k] = max(self._gneg[k - 1] + log_neg, 0)

            # Check for threshold crossing
            if self._gpos[k] > self.threshold or self._gneg[k] > self.threshold:
                if self._gpos[k] > self.threshold:
                    jump = self._find_jump_location(self._cpos, k)
                    self._process_jump(jump)

                if self._gneg[k] > self.threshold:
                    jump = self._find_jump_location(self._cneg, k)
                    self._process_jump(jump)

                # Reset after detection
                self._anchor = k
                self._reset_arrays()
                self._stats.reset(data[k])

        # Add final edge
        self._edges.append(length)
        return len(self._edges) - 1

    def detect(
        self,
        data: npt.NDArray[np.floating],
        max_states: int = -1,
    ) -> CUSUMResultDict:
        """
        Detect change points in the data.

        Parameters
        ----------
        data : npt.NDArray[np.floating]
            Input time series data.
        max_states : int, optional
            Maximum allowed states. If exceeded, threshold is relaxed.
            Use -1 for no limit, 0 for single state.

        Returns
        -------
        CUSUMResultDict
            Detection results with keys: nStates, starts, threshold, stepsize.
        """
        if max_states == 0:
            return _create_single_state_result(len(data), self.threshold, self.stepsize)

        # Adaptive detection with threshold relaxation
        relaxation_factor = 1.1
        current_threshold = self.threshold
        current_stepsize = self.stepsize

        while True:
            self._reset()
            self.threshold = current_threshold
            self.stepsize = current_stepsize

            n_states = self._detect_single_pass(data)

            if max_states < 0 or n_states <= max_states:
                break

            # Relax parameters if too many states detected
            current_threshold *= relaxation_factor
            current_stepsize *= relaxation_factor

        edges = np.array(self._edges, dtype=np.int64)
        assert n_states == len(edges) - 1

        return {
            "nStates": n_states,
            "starts": edges,
            "threshold": self.threshold,
            "stepsize": self.stepsize,
        }


def detect_cusumv2(
    data0: npt.NDArray[np.floating],
    basesd: float,
    dt: Optional[float] = None,
    threshhold: float = 10,
    stepsize: float = 3,
    minlength: int = 1000,
    maxstates: int = -1,
    moving_oneside_window: int = 0,
) -> CUSUMResultDict:
    """
    Detect change points in time series data using CUSUM algorithm.

    This function identifies abrupt changes (jumps) in the data, commonly used
    for analyzing nanopore or ion channel recordings.

    Parameters
    ----------
    data0 : npt.NDArray[np.floating]
        Input time series data.
    basesd : float
        Baseline standard deviation (noise level).
    dt : float, optional
        Time step (currently unused, kept for API compatibility).
    threshhold : float, default=10
        Detection threshold. Higher values reduce sensitivity.
    stepsize : float, default=3
        Expected jump size in units of basesd.
    minlength : int, default=1000
        Minimum number of samples between detected changes.
    maxstates : int, default=-1
        Maximum number of states. If exceeded, threshold is relaxed.
        Use 0 for single state, -1 for unlimited.
    moving_oneside_window : int, default=0
        Window size for preprocessing filters. Use 0 to skip preprocessing.

    Returns
    -------
    CUSUMResultDict
        Detection results containing:
        - nStates: Number of detected states
        - starts: Array of state boundary indices
        - threshold: Final threshold used
        - stepsize: Final stepsize used
    """
    # Handle single-state case
    if maxstates == 0:
        return _create_single_state_result(len(data0), threshhold, stepsize)

    # Preprocess data
    data = _preprocess_data(data0, moving_oneside_window)

    if data is None:
        # Data too short for preprocessing
        return _create_single_state_result(len(data0), threshhold, stepsize)

    assert len(data) == len(data0)

    # Run detection
    detector = CUSUMDetector(
        base_sd=basesd,
        threshold=threshhold,
        stepsize=stepsize,
        min_length=minlength,
    )

    return detector.detect(data, max_states=maxstates)
