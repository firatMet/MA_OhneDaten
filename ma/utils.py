import pathlib
from datetime import datetime as dt
from typing import Literal
import pandas as pd
import numpy as np
import re
import pywt
from scipy.signal import butter, filtfilt, medfilt
from scipy.fft import fft, fftfreq

DATA_DIR = pathlib.Path("data/Unovis_cumumovi2023/").resolve()
SAMPLING_FREQ = 128


def load_all_reference_signals(
    signal_type: Literal["ecg_ref", "resp_ref"], class_name: Literal["HR", "RR"]
):
    ref_data = pd.DataFrame({})
    for parti_no in range(1, 21):
        data = load_participant_data(
            participant_number=parti_no,
            signal_names=[signal_type],
            bandpassed=False,
            normalized=False,
            segmented=False,
            class_name=class_name,
            load_labels=False,
            time_offset=True,
        )
        data["parti"] = parti_no
        ref_data = pd.concat([ref_data, data])
    return ref_data


def load_all_signals(
    signal_type: Literal["ECG", "PPG", "SCG", "MI"],
    bandpassed: bool = True,
    normalized: bool = False,
    class_name: Literal["HR", "RR"] = "HR",
) -> list[pd.DataFrame]:
    assert isinstance(signal_type, str)
    signal_names = get_signal_names(signal_type=signal_type)

    all_data = []
    for parti_no in range(1, 21):
        data, labels = load_participant_data(
            participant_number=parti_no,
            signal_names=signal_names,
            bandpassed=bandpassed,
            normalized=normalized,
            segmented=True,
            class_name=class_name,
            load_labels=True,
            time_offset=True,
        )
        data = data[:-1]
        data = put_metadata_to_segments(segments=data, labels=labels, parti_no=parti_no)
        all_data.extend(data)

    return all_data


def load_participant_data(
    participant_number: int,
    signal_names: list[str] = None,
    bandpassed: bool = False,
    normalized: bool = False,
    segmented: bool = False,
    class_name: Literal["HR", "RR"] = "HR",
    load_labels: bool = False,
    time_offset: bool = False,
) -> pd.DataFrame:
    assert class_name in ["HR", "RR"], "Please give a valid class name, HR or RR"

    filename = "participant" + str(participant_number) + ".csv"
    if signal_names and "time" not in signal_names:
        signal_names += ["time"]
    data = pd.read_csv(DATA_DIR / filename, usecols=signal_names)

    def parser(x):
        return dt.fromtimestamp(float(x))

    data["time"] = data["time"].apply(parser)
    data.set_index("time", inplace=True)

    if time_offset:
        data.index = (data.index - data.index[0]).total_seconds()

    if bandpassed:
        data = _bandpass_filter(df_participant=data, class_name=class_name)

    if normalized:
        data = _normalize_data(df_participant=data)

    if segmented:
        data = _segment_participant_data(df_participant=data, class_name=class_name)

    if load_labels:
        signal_names.remove("time")
        labels = {signal_name: pd.DataFrame({}) for signal_name in signal_names}
        for signal_name in signal_names:
            label = _load_signal_label(
                participant_number=participant_number,
                signal_name=signal_name,
                class_name=class_name,
            )
            labels[signal_name] = label

        return data, labels
    return data


def _load_signal_label(
    participant_number: int, signal_name: str, class_name: str
) -> pd.DataFrame:
    signal_name = signal_name.upper()
    signal_type = re.split(r"\d", signal_name)[0]
    match signal_type:
        # case "ECG":
        #     signal_name = "c" + signal_name
        case "PPG":
            signal_name = "r" + signal_name
        case "SCG":
            signal_name = signal_name[:4] + signal_name[4].lower()

    folder_name = "labeled_data/participant" + str(participant_number)
    file_name = signal_name + "_" + class_name + ".xlsx"
    path = DATA_DIR.parent / folder_name / file_name

    labels = pd.read_excel(path)
    labels["signal"] = [signal_name] * len(labels)

    return labels


def put_metadata_to_segments(
    segments: list[pd.DataFrame], labels: dict[pd.DataFrame], parti_no: int
) -> list[pd.DataFrame]:
    for segment_no, segment in enumerate(segments):
        for column in segment.columns:
            segment.attrs[f"{column}_sq"] = labels[column]["signalquality"][segment_no]
            segment.attrs[f"{column}_ma"] = labels[column]["artifacts"][segment_no]
        segment.attrs["signal"] = f"parti: {parti_no} | segment: {segment_no + 1}"

    return segments


def _normalize_data(df_participant: pd.DataFrame) -> pd.DataFrame:
    normalized_df = (df_participant - df_participant.mean()) / df_participant.std()
    return normalized_df


def _bandpass_filter(df_participant: pd.DataFrame, class_name: str) -> pd.DataFrame:
    # Get the filters for each signal type:
    match class_name:
        case "HR":
            b_ecg, a_ecg = butter(2, [1, 40], btype="bandpass", fs=SAMPLING_FREQ)
            b_ppg, a_ppg = butter(2, [0.4, 10], btype="bandpass", fs=SAMPLING_FREQ)
        case "RR":
            b_ppg, a_ppg = butter(2, [0.1, 0.5], btype="bandpass", fs=SAMPLING_FREQ)
            b_scg, a_scg = butter(2, [0.1, 1], btype="bandpass", fs=SAMPLING_FREQ)
            b_mi, a_mi = butter(2, 0.1, btype="high", fs=SAMPLING_FREQ)

    # Filter signals
    df_filtered = df_participant.copy()
    for column_name in df_filtered.columns:
        signal_type = re.split(r"\d", column_name)[0]
        if signal_type not in ["ecg", "ppg", "mi", "scg"]:
            continue
        match signal_type:
            case "ecg":
                b, a = b_ecg, a_ecg
            case "ppg":
                b, a = b_ppg, a_ppg
            case "scg":
                b, a = b_scg, a_scg
            case "mi":
                b, a = b_mi, a_mi
        df_filtered[column_name] = filtfilt(b, a, df_filtered[column_name])

        if signal_type == "mi":  # also apply median filtering to mi signals:
            df_filtered[column_name] = medfilt(
                volume=df_filtered[column_name], kernel_size=65
            )

        if signal_type == "ppg":  # also apply median filtering to mi signals:
            df_filtered[column_name] = medfilt(
                volume=df_filtered[column_name], kernel_size=25
            )
    return df_filtered


def fft_filter(df_segment: pd.DataFrame) -> pd.DataFrame:
    N = len(df_segment)
    T = 1 / SAMPLING_FREQ
    df = df_segment.copy()

    # Filter signals
    xf = fftfreq(N, T)[: N // 2]
    df_filtered = pd.DataFrame({"freq": xf})
    for column_name in df.columns:
        signal_type = re.split(r"\d", column_name)[0]
        if signal_type not in ["ecg", "ppg", "mi", "scg"]:
            continue
        # Apply FFT to the PPG signal
        fft_values = fft(df[column_name].values)
        # Single Sided
        fft_amplitude = 2.0 / N * np.abs(fft_values[: N // 2])
        df_filtered[column_name] = fft_amplitude

    df_filtered.set_index("freq", inplace=True)

    return df_filtered


def wavelet_transform(df_segment: pd.DataFrame) -> pd.DataFrame:
    wavelet = "db1"
    wavelet_transformed_data = {}
    for column in df_segment.columns:
        # Perform DWT on the column
        coeffs = pywt.dwt(df_segment[column], wavelet)

        # Save the approximation and detail coefficients
        wavelet_transformed_data[column] = coeffs[0]

    # Output wavelet transformed data
    truncated_index = df_segment.index[: len(wavelet_transformed_data[column])]
    return pd.DataFrame(wavelet_transformed_data, index=truncated_index)


def _segment_participant_data(
    df_participant: pd.DataFrame, class_name: Literal["HR", "RR"]
) -> list[pd.DataFrame]:
    """
    Segments data in 2 seconds (for HR) or 10 seconds (for RR) of intervalls.

    Parameters
    ----------
    df_participant : pd.DataFrame
        Data to be segmented.

    Returns
    -------
    list[pd.DataFrame]
        List of 2 seconds long dataframes.
    """
    match class_name:
        case "HR":
            chunk = 256
        case "RR":
            chunk = 1280
        case _:
            raise ValueError(
                f"""Please be sure that the given class name is valid: {class_name}
                                 Valid class names are: 'HR' & 'RR'."""
            )
    df_chunks = [
        df_participant.iloc[i : i + chunk] for i in range(0, len(df_participant), chunk)
    ]
    return df_chunks


def get_signal_names(signal_type: Literal["ECG", "PPG", "SCG", "MI"]) -> list[str]:
    match signal_type:
        case "ECG":
            signal_name = ["ecg1", "ecg2", "ecg3", "ecg4"]
        case "PPG":
            signal_name = ["ppg1", "ppg2", "ppg3", "ppg4"]
        case "MI":
            signal_name = ["mi1", "mi2", "mi3", "mi4"]
        case "SCG":
            signal_name = [
                "scg1x",
                "scg1y",
                "scg1z",
                "scg2x",
                "scg2y",
                "scg2z",
                "scg3x",
                "scg3y",
                "scg3z",
                "scg4x",
                "scg4y",
                "scg4z",
            ]
        case _:
            raise ValueError(f"INVALID SIGNAL TYPE: {signal_type}")
    return signal_name
