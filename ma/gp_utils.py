from ma.utils import load_all_signals
from typing import Literal
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from tensorflow.keras.models import model_from_json
import joblib
from scipy.ndimage import gaussian_filter1d


SEGMENT_LENGTH = 256
MERGE_TOL = 0.3  # seconds
RESAMPLE_FREQ = 0.5  #


def load_deep_model(model_files):
    """Load a deep learning model from JSON and weights files."""
    from ma.gp import MODEL_PATH

    # Load model architecture
    with open(MODEL_PATH / model_files[0], "r") as json_file:
        loaded_network_json = json_file.read()
    model = model_from_json(loaded_network_json)

    # Load model weights
    model.load_weights(MODEL_PATH / model_files[1])

    # Recompile the model (required for training or evaluation)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def load_sqi_model(model_files):
    from ma.gp import MODEL_PATH

    feature_path = [element for element in model_files if "features" in element]
    sqi_model_path = [element for element in model_files if "features" not in element]

    return (
        joblib.load(MODEL_PATH / sqi_model_path[0]),
        joblib.load(MODEL_PATH / feature_path[0]),
    )


def _get_error_metrics(diffs: list[float]) -> dict[str, float]:
    # Mean Squared Error (MSE)
    mse = np.mean(diffs**2)
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(diffs))
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    # Median Absolute Error
    median_absolute_error = np.median(np.abs(diffs))

    return {"mse": mse, "mae": mae, "rmse": rmse, "med_abs_err": median_absolute_error}


def calculate_hr_error_ma_intervals(
    df_parti_hr: pd.DataFrame,
    df_ref_hr: pd.DataFrame,
    ma_indexes=list[int, int],
    recovered: bool = False,
) -> dict[str, float]:
    ref_signal = df_ref_hr.columns[0]  # hr or rr
    if recovered:
        diffs = []
        for ma_index in ma_indexes:
            diff = (
                df_parti_hr.loc[ma_index][ref_signal]
                - df_ref_hr.loc[ma_index][ref_signal]
            )
            diff = diff.values[:-1]
            diffs.extend(diff)
        diffs = np.array(diffs)
    else:
        diffs = []
        for ma_index in ma_indexes:
            ma_ts = df_parti_hr.loc[ma_index[0] : ma_index[-1]]
            idx_start = np.argmin(abs(df_ref_hr.index - ma_ts.index[0]))
            idx_end = np.argmin(abs(df_ref_hr.index - ma_ts.index[-1]))
            ma_ts = df_ref_hr.index[idx_start : idx_end + 1]
            # Piece-wise linear interpolation:
            interpolated_values = []
            for ts in ma_ts:
                try:
                    idx = np.where(df_parti_hr.index - ts > 0)[0][0]
                except:
                    idx = len(df_parti_hr) - 1
                start_ts = df_parti_hr.index[idx - 1]
                stop_ts = df_parti_hr.index[idx]
                x = [start_ts, stop_ts]
                y = [
                    df_parti_hr.loc[start_ts][ref_signal],
                    df_parti_hr.loc[stop_ts][ref_signal],
                ]
                interpolated_values.append(np.interp(ts, x, y))
            diff = df_ref_hr.loc[ma_ts][ref_signal].values - interpolated_values
            diffs.extend(diff)
        diffs = np.array(diffs)

    return _get_error_metrics(diffs)


def calculate_hr_error_overall(
    df_parti_hr: pd.DataFrame, df_ref_hr: pd.DataFrame
) -> dict[str, float]:
    diffs = []
    ref_signal = df_ref_hr.columns[0]  # hr or rr
    for row in df_ref_hr.iterrows():
        ref_value = row[1][ref_signal]
        ref_time = row[0]

        # Find the two closest timestamps in df_recovered
        times_below = df_parti_hr[df_parti_hr.index <= ref_time]
        times_above = df_parti_hr[df_parti_hr.index > ref_time]

        if times_below.empty or times_above.empty:
            continue

        # Get the closest times
        t_below = times_below.index[-1]
        hr_below = df_parti_hr.loc[t_below, ref_signal]
        if hr_below.size > 1:
            hr_below = hr_below.iloc[0]

        t_above = times_above.index[0]
        hr_above = df_parti_hr.loc[t_above, ref_signal]
        if hr_above.size > 1:
            hr_above = hr_above.iloc[0]

        # Perform linear interpolation
        interpolated_hr = hr_below + (hr_above - hr_below) * (
            (ref_time - t_below) / (t_above - t_below)
        )

        # Calculate the difference
        difference = interpolated_hr - ref_value
        diffs.append(difference)

    return _get_error_metrics(np.array(diffs))


def calculate_hr_or_rr(
    df_parti: pd.DataFrame,
    peaks: list[list[int]],
    class_name: Literal["HR", "RR"],
) -> pd.DataFrame:
    ts = []
    rates = []
    sq_labels = []
    ma_labels = []
    last_peak = None
    flatten_factor = 256 if class_name == "HR" else 1280
    peaks_flattened = [
        value + flatten_factor * idx
        for idx, sublist in enumerate(peaks[0])
        for value in sublist
    ]
    sq_labels_flattened = [
        label
        for idx, label in enumerate(peaks[1])
        for _ in peaks[0][idx]  # Repeat according to length of each sublist in peaks[1]
    ]

    ma_labels_flattened = [
        label
        for idx, label in enumerate(peaks[2])
        for _ in peaks[0][idx]  # Repeat according to length of each sublist in peaks[2]
    ]

    for peak, sq_label, ma_label in zip(
        np.arange(0, len(peaks_flattened)), sq_labels_flattened, ma_labels_flattened
    ):
        if not last_peak:
            last_peak = df_parti.index[peaks_flattened[peak]]
            last_label_sq = sq_label
            last_label_ma = ma_label
            continue

        # Calculate the HR/RR Values:
        time_diff = df_parti.index[peaks_flattened[peak]] - last_peak
        heart_rate = 60 / time_diff
        ts.append((time_diff / 2) + last_peak)
        rates.append(heart_rate)

        # Get the HR/RR Label:
        hr_rr_label_sq = 0
        if sq_label == 1 or last_label_sq == 1:
            hr_rr_label_sq = 1
        if sq_label == 2 or last_label_sq == 2:
            hr_rr_label_sq = 2
        if sq_label == -1 or last_label_sq == -1:
            hr_rr_label_sq = -1
        sq_labels.append(hr_rr_label_sq)

        hr_rr_label_ma = 0
        if ma_label == 1 or last_label_ma == 1:
            hr_rr_label_ma = 1
        if ma_label == -1 or last_label_ma == -1:
            hr_rr_label_ma = -1
        ma_labels.append(hr_rr_label_ma)

        # Update iteration variables:
        last_peak = df_parti.index[peaks_flattened[peak]]
        last_label_sq = sq_label
        last_label_ma = ma_label

    match class_name:
        case "HR":
            df_rates = pd.DataFrame(
                {"hr": rates, "sq_labels": sq_labels, "ma_labels": ma_labels}, index=ts
            )
        case "RR":
            df_rates = pd.DataFrame(
                {"rr": rates, "sq_labels": sq_labels, "ma_labels": ma_labels}, index=ts
            )
        case _:
            raise ValueError("Please give a valid class name HR or RR !")

    return df_rates


def mi_peak_detection(
    df_segmented: list[pd.DataFrame],
    signal_name: str,
    distance=200,
    dynamic_height_multiplier=0.001,
    prominence=0.001,
    sigma=10,
) -> list[int]:
    all_peaks = []
    peak_labels_sq = []
    peak_labels_ma = []

    last_peak_from_previous_segment = None

    for segment in df_segmented:
        segment = segment[signal_name]
        # Apply Gaussian smoothing
        smoothed_signal = gaussian_filter1d(segment, sigma=sigma)

        # Calculate dynamic height threshold
        dynamic_height = np.mean(smoothed_signal) + dynamic_height_multiplier * np.std(
            smoothed_signal
        )

        # Detect peaks across the entire combined segment
        peaks, _ = find_peaks(
            smoothed_signal,
            distance=distance,
            height=dynamic_height,
            prominence=prominence,
        )

        peaks, last_peak_from_previous_segment = (
            _check_distance_on_consecutive_segments(
                last_peak_from_prev_segment=last_peak_from_previous_segment,
                peaks=peaks.tolist(),
                len_segment=len(segment),
                distance=distance,
            )
        )

        labels = _get_peak_labels(segment, signal_name)
        peak_labels_sq.append(labels["sq"])
        peak_labels_ma.append(labels["ma"])
        
        all_peaks.append(peaks)

    return all_peaks, peak_labels_sq, peak_labels_ma


def scg_peak_detection(
    df_segmented: list[pd.DataFrame],
    signal_name: str,
    distance=200,
    dynamic_height_multiplier=0.3,
    prominence=0.5,
    sigma=50,
) -> list[int]:
    all_peaks = []
    peak_labels_sq = []
    peak_labels_ma = []

    last_peak_from_previous_segment = None

    for segment in df_segmented:
        segment = segment[signal_name]
        # Apply Gaussian smoothing
        smoothed_signal = gaussian_filter1d(segment, sigma=sigma)

        # Calculate dynamic height threshold
        dynamic_height = np.mean(smoothed_signal) + dynamic_height_multiplier * np.std(
            smoothed_signal
        )

        # Detect peaks across the entire combined segment
        peaks, _ = find_peaks(
            smoothed_signal,
            distance=distance,
            height=dynamic_height,
            prominence=prominence,
        )

        peaks, last_peak_from_previous_segment = (
            _check_distance_on_consecutive_segments(
                last_peak_from_prev_segment=last_peak_from_previous_segment,
                peaks=peaks.tolist(),
                len_segment=len(segment),
                distance=distance,
            )
        )

        labels = _get_peak_labels(segment, signal_name)
        peak_labels_sq.append(labels["sq"])
        peak_labels_ma.append(labels["ma"])

        all_peaks.append(peaks)

    return all_peaks, peak_labels_sq, peak_labels_ma


def ppg_peak_detection(
    df_segmented: list[pd.DataFrame],
    signal_name: str,
    distance: float = 40.0,
    dynamic_height_multiplier: float = 0.25,
    prominence: float = 0.2,
    sigma=6.2,
) -> list[int]:
    all_peaks = []
    peak_labels_sq = []
    peak_labels_ma = []
    last_peak_from_previous_segment = None

    all_peaks = []
    for segment in df_segmented:
        segment = segment[signal_name]
        # Apply Gaussian smoothing
        smoothed_signal = gaussian_filter1d(segment, sigma=sigma)
        # Detect peaks with a minimum distance, height, and prominence
        dynamic_height = np.mean(smoothed_signal) + dynamic_height_multiplier * np.std(
            smoothed_signal
        )
        peaks, _ = find_peaks(
            smoothed_signal,
            distance=distance,
            height=dynamic_height,
            prominence=prominence,
        )

        # It finds the maximum of the peak as default, so modify it
        peak_start_indices = []
        for peak in peaks:
            # Traverse backwards to find where the slope changes
            for i in range(peak, 0, -1):
                if (segment.iloc[i] - segment.iloc[i - 1]) > 0.075:
                    peak_start_indices.append(i)
                    break

        peak_start_indices, last_peak_from_previous_segment = (
            _check_distance_on_consecutive_segments(
                last_peak_from_prev_segment=last_peak_from_previous_segment,
                peaks=peak_start_indices,
                len_segment=len(segment),
                distance=distance,
            )
        )

        labels = _get_peak_labels(segment, signal_name)
        peak_labels_sq.append(labels["sq"])
        peak_labels_ma.append(labels["ma"])

        all_peaks.append(peak_start_indices)
        
    return all_peaks, peak_labels_sq, peak_labels_ma


def ecg_r_peak_detection(
    df_segmented: list[pd.DataFrame], signal_name: str, height_threshold: float = 0.5
) -> list[int]:
    """Applies Pan-Tompkins R-Peak Detection to the Segmented ECG Data.

    Parameters
    ----------
    df_segmented : list[pd.DataFrame]
        Segmented ECG Data
    height_threshold : float, optional
        Threshold for peaks, by default 0.5
    """

    def differentiate(signal: np.ndarray):
        diff_signal = np.diff(signal)
        return diff_signal

    def square(signal: np.ndarray):
        squared_signal = signal**2
        return squared_signal

    def moving_window_integration(signal: np.ndarray, window_size: int):
        integrated_signal = np.convolve(
            signal, np.ones(window_size) / window_size, mode="same"
        )
        return integrated_signal

    all_peaks = []
    peak_labels_sq = []
    peak_labels_ma = []
    last_peak_from_previous_segment = None

    for segment in df_segmented:
        segment = segment[signal_name]
        fs = 1 / ((segment.index[1] - segment.index[0]))
        ecg_signal = segment.values
        # Differentiation
        diff_signal = differentiate(ecg_signal)

        # Squaring
        squared_signal = square(diff_signal)

        # Moving window integration
        window_size = int(0.12 * fs)
        integrated_signal = moving_window_integration(squared_signal, window_size)

        # Find peaks
        distance = int(0.5 * fs)
        ht = height_threshold * max(integrated_signal)
        peaks, _ = find_peaks(integrated_signal, distance=distance, height=ht)
        corrected_peaks = []
        for peak in peaks:
            # Define a window around the peak to find a more precise maximum
            start = max(peak - 10, 0)
            end = min(peak + 10, len(segment))
            segment_peripheral = segment.iloc[start:end]

            # Find the local maximum within this peripheral window
            local_max_index = np.argmax(segment_peripheral)

            # Correct the peak position relative to the original peak
            corrected_peak = start + local_max_index
            corrected_peaks.append(corrected_peak)

        corrected_peaks, last_peak_from_previous_segment = (
            _check_distance_on_consecutive_segments(
                last_peak_from_prev_segment=last_peak_from_previous_segment,
                peaks=corrected_peaks,
                len_segment=len(segment),
                distance=distance,
            )
        )

        labels = _get_peak_labels(segment, signal_name)
        peak_labels_sq.append(labels["sq"])
        peak_labels_ma.append(labels["ma"])

        all_peaks.append(corrected_peaks)

    return all_peaks, peak_labels_sq, peak_labels_ma


def _get_peak_labels(segment: pd.DataFrame, signal_name: str) -> dict[str, int]:
    if signal_name in ("ecg_ref", "resp_ref"):
        return {"sq": 0, "ma": 0}
    else:
        return {
            "sq": segment.attrs[f"{signal_name}_sq"],
            "ma": segment.attrs[f"{signal_name}_ma"],
        }

def _check_distance_on_consecutive_segments(
    last_peak_from_prev_segment: int,
    peaks: list[int],
    len_segment: int,
    distance: float,
):
    # Check out the distance between last peak of n-th & first peak of n+1 th segment
    if len(peaks) > 0:
        if last_peak_from_prev_segment is not None:
            first_peak_from_current_segment = peaks[0]
            if (
                first_peak_from_current_segment
                - last_peak_from_prev_segment
                + len_segment
                < distance
            ):
                try:
                    peaks.pop(0)
                except:
                    peaks.pop(0)
        last_peak_from_prev_segment = peaks[-1] if len(peaks) > 0 else None
    return peaks, last_peak_from_prev_segment


def combine_ma_intervals(ma_indexes: list[list[int, int]]) -> list[list[int, int]]:
    # Combine the consecutive MA segments:
    intervals_to_delete = []
    for interval_no, ma_index in enumerate(ma_indexes):
        if interval_no == 0:
            stop_idx = ma_index[-1]
            continue

        start_idx = ma_index[0]
        # Consecutive MA segments:
        if start_idx - stop_idx < 0:
            ma_index[0] = ma_indexes[interval_no - 1][0]
            intervals_to_delete.append(interval_no - 1)
        # Update stop_idx:
        stop_idx = ma_index[-1]

    ma_indexes = [
        item
        for index, item in enumerate(ma_indexes)
        if index not in intervals_to_delete
    ]

    return ma_indexes
