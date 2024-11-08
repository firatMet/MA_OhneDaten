import pandas as pd
import numpy as np

from ma.utils import load_participant_data, put_metadata_to_segments
from ma.gp_utils import (
    combine_ma_intervals,
    calculate_hr_error_ma_intervals,
    calculate_hr_error_overall,
    calculate_hr_or_rr,
    ecg_r_peak_detection,
    ppg_peak_detection,
    scg_peak_detection,
    mi_peak_detection,
)
from ma.gp_visu import plot_signals_with_hr
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
from abc import ABC

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RationalQuadratic,
    ConstantKernel as C,
    RBF,
    ExpSineSquared,
    Matern,
)
from sklearn.preprocessing import StandardScaler

HR_SIGNALS = [
    "ecg1",
    "ecg2",
    "ecg3",
    "ecg4",
    "ppg1",
    "ppg2",
    "ppg3",
    "ppg4",
]
RR_SIGNALS = [
    # "ecg1",
    # "ecg2",
    # "ecg3",
    # "ecg4",
    # "ppg1",
    # "ppg2",
    # "ppg3",
    # "ppg4",
    "scg1x",
    "scg2x",
    "scg3x",
    "scg4x",
    "scg1y",
    "scg2y",
    "scg3y",
    "scg4y",
    "scg1z",
    "scg2z",
    "scg3z",
    "scg4z",
    "mi1",
    "mi2",
    "mi3",
    "mi4",
]

peak_detection_func_mapping = {
    "ECG": ecg_r_peak_detection,
    "PPG": ppg_peak_detection,
    "SCG": scg_peak_detection,
    "MI": mi_peak_detection,
}


class MTGP:
    def __init__(self, parti_no: int) -> None:
        self.parti_no = parti_no
        self._initialize_participant_data()
        self._detect_peaks()
        self.hr_merged = pd.DataFrame({})
        self.rr_merged = pd.DataFrame({})
        self._calculate_hr_rr()

        self.hr_fused = pd.DataFrame({})
        self.rr_fused = pd.DataFrame({})
        self._fuse_hr_rr()

    def _fuse_hr_rr(self) -> None:
        # HR:
        self.hr_fused = self.hr_merged[["hr_ref", "sq_labels_ref", "ma_labels_ref"]]
        hr_values = []
        sq_labels = []
        ma_labels = []
        from_which_signal = []
        for row in self.hr_merged.iterrows():
            index = row[0]
            signal = self._find_signal_to_fuse_from(row, HR_SIGNALS)
            hr_values.append(self.hr_merged[f"hr_{signal}"].loc[index])
            sq_labels.append(self.hr_merged[f"sq_labels_{signal}"].loc[index])
            ma_labels.append(self.hr_merged[f"ma_labels_{signal}"].loc[index])
            from_which_signal.append(signal)

        self.hr_fused["hr_fused"] = hr_values
        self.hr_fused["sq_labels_fused"] = sq_labels
        self.hr_fused["ma_labels_fused"] = ma_labels
        self.hr_fused["from"] = from_which_signal

        # RR:
        self.rr_fused = self.rr_merged[["rr_ref", "sq_labels_ref", "ma_labels_ref"]]
        rr_values = []
        sq_labels = []
        ma_labels = []
        from_which_signal = []
        for row in self.rr_merged.iterrows():
            index = row[0]
            signal = self._find_signal_to_fuse_from(row, RR_SIGNALS)
            rr_values.append(self.rr_merged[f"rr_{signal}"].loc[index])
            sq_labels.append(self.rr_merged[f"sq_labels_{signal}"].loc[index])
            ma_labels.append(self.rr_merged[f"ma_labels_{signal}"].loc[index])
            from_which_signal.append(signal)

        self.rr_fused["rr_fused"] = rr_values
        self.rr_fused["sq_labels_fused"] = sq_labels
        self.rr_fused["ma_labels_fused"] = ma_labels
        self.rr_fused["from"] = from_which_signal

    def _find_signal_to_fuse_from(self, row, signals) -> str:
        conditions = [  # is the order ok?
            (0, 0),  # (sq=0, ma=0)
            (1, 0),  # (sq=1, ma=0)
            (0, 1),  # (sq=0, ma=1)
            (1, 1),  # (sq=1, ma=1)
            (2, 0),  # (sq=2, ma=0)
            (2, 1),  # (sq=2, ma=1)
        ]

        first_match = None
        for sq_val, ma_val in conditions:
            for signal in signals:
                    # Check if both conditions are met for this signal
                    if (
                        row[1][f"sq_labels_{signal}"] == sq_val
                        and row[1][f"ma_labels_{signal}"] == ma_val
                    ):
                        first_match = signal
                        break
            if first_match:
                break

        # If no match was found with conditions, take the first signal
        if not first_match:
            first_match = signals[0]

        return first_match

    def _merge_hr_rr(self, class_name: str) -> None:
        signals = HR_SIGNALS if class_name == "HR" else RR_SIGNALS
        df_ref = self.hr_rr_ref[class_name]
        df_merged = df_ref.copy()
        for signal in signals:
            df = self.hr_rr_parti[class_name][signal]
            df = df[~df.index.duplicated(keep="first")]
            interpolated_values = []
            sq_labels = []
            ma_labels = []
            for row in df_ref.iterrows():
                ref_index = row[0]
                parti_index = np.where((df.index - ref_index) < 0)[0]
                if len(parti_index) == 0:  # First index
                    value = df[class_name.lower()].values[0]
                    sq_labels.append(df["sq_labels"].values[0])
                    ma_labels.append(df["ma_labels"].values[0])
                else:
                    # Linear Interpolation:
                    x1 = df.index[parti_index[-1]]
                    y1 = df[class_name.lower()].iloc[parti_index[-1]]
                    if parti_index[-1] + 1 == len(df):  # Last index:
                        value = df[class_name.lower()].values[-1]
                        sq_label2 = sq_label1 = df["sq_labels"].values[-1]
                        ma_label2 = ma_label1 = df["ma_labels"].values[-1]
                    else:  # Intermediate Indexes:
                        x2 = df.index[parti_index[-1] + 1]
                        y2 = df[class_name.lower()].iloc[parti_index[-1] + 1]
                        if x1 == x2:
                            value = y2
                        else:
                            value = y1 + (y2 - y1) * ((ref_index - x1) / (x2 - x1))

                        sq_label1 = df["sq_labels"].loc[x1]
                        sq_label2 = df["sq_labels"].loc[x2]
                        ma_label1 = df["ma_labels"].loc[x1]
                        ma_label2 = df["ma_labels"].loc[x2]

                    sq_label = 0
                    if sq_label1 == 1 or sq_label2 == 1:
                        sq_label = 1
                    if sq_label1 == 2 or sq_label2 == 2:
                        sq_label = 2
                    if sq_label1 == -1 or sq_label2 == -1:
                        sq_label = -1

                    ma_label = 0
                    if ma_label1 == 1 or ma_label2 == 1:
                        ma_label = 1
                    if ma_label1 == 2 or ma_label2 == 2:
                        ma_label = 2
                    if ma_label1 == -1 or ma_label2 == -1:
                        ma_label = -1

                    sq_labels.append(sq_label)
                    ma_labels.append(ma_label)
                interpolated_values.append(value)
            df_merged[f"{class_name.lower()}_{signal}"] = interpolated_values
            df_merged[f"sq_labels_{signal}"] = sq_labels
            df_merged[f"ma_labels_{signal}"] = ma_labels

        return df_merged

    def _calculate_hr_rr(self) -> None:
        self.hr_rr_parti = {"HR": {}, "RR": {}}
        self.hr_rr_ref = {"HR": [], "RR": []}

        for signal in HR_SIGNALS:
            self.hr_rr_parti["HR"][signal] = calculate_hr_or_rr(
                df_parti=self.df_parti_hr[signal],
                peaks=self.peaks_parti["HR"][signal],
                class_name="HR",
            )

        for signal in RR_SIGNALS:
            self.hr_rr_parti["RR"][signal] = calculate_hr_or_rr(
                df_parti=self.df_parti_rr[signal],
                peaks=self.peaks_parti["RR"][signal],
                class_name="RR",
            )

        self.hr_rr_ref["HR"] = calculate_hr_or_rr(
            df_parti=self.df_ref_hr, peaks=self.peaks_ref["HR"], class_name="HR"
        )

        self.hr_rr_ref["RR"] = calculate_hr_or_rr(
            df_parti=self.df_ref_hr, peaks=self.peaks_ref["RR"], class_name="RR"
        )

        self.hr_merged = self._merge_hr_rr(class_name="HR")
        self.hr_merged.rename(
            columns={
                "hr": "hr_ref",
                "sq_labels": "sq_labels_ref",
                "ma_labels": "ma_labels_ref",
            },
            inplace=True,
        )
        self.rr_merged = self._merge_hr_rr(class_name="RR")
        self.rr_merged.rename(
            columns={
                "rr": "rr_ref",
                "sq_labels": "sq_labels_ref",
                "ma_labels": "ma_labels_ref",
            },
            inplace=True,
        )

    def _detect_peaks(self) -> None:
        self.peaks_parti = {"HR": {}, "RR": {}}
        self.peaks_ref = {"HR": [], "RR": []}
        for signal in HR_SIGNALS:
            self.current_signal_name = signal
            func_peak_detection = peak_detection_func_mapping.get(self.signal_type)
            self.peaks_parti["HR"][signal] = func_peak_detection(
                self.df_parti_segmented_hr, signal_name=signal
            )

        for signal in RR_SIGNALS:
            self.current_signal_name = signal
            func_peak_detection = peak_detection_func_mapping.get(self.signal_type)
            self.peaks_parti["RR"][signal] = func_peak_detection(
                self.df_parti_segmented_rr, signal_name=signal
            )

        self.peaks_ref["HR"] = ecg_r_peak_detection(
            self.df_ref_segmented_hr, signal_name="ecg_ref"
        )
        self.peaks_ref["RR"] = scg_peak_detection(
            self.df_ref_segmented_rr, signal_name="resp_ref"
        )

    def _initialize_participant_data(self) -> None:
        # HR - Parti:
        self.df_parti_hr = load_participant_data(
            participant_number=self.parti_no,
            signal_names=HR_SIGNALS,
            bandpassed=True,
            normalized=False,
            segmented=False,
            class_name="HR",
            load_labels=False,
            time_offset=True,
        )

        parti_data, parti_labels = load_participant_data(
            participant_number=self.parti_no,
            signal_names=HR_SIGNALS,
            bandpassed=True,
            normalized=False,
            segmented=True,
            class_name="HR",
            load_labels=True,
            time_offset=True,
        )
        self.df_parti_segmented_hr = put_metadata_to_segments(
            segments=parti_data, labels=parti_labels, parti_no=self.parti_no
        )[:-1]

        # HR - Reference:
        self.df_ref_hr = load_participant_data(
            participant_number=self.parti_no,
            signal_names=["ecg_ref"],
            segmented=False,
            time_offset=True,
            class_name="HR",
        )

        self.df_ref_segmented_hr = load_participant_data(
            participant_number=self.parti_no,
            signal_names=["ecg_ref"],
            segmented=True,
            time_offset=True,
            class_name="HR",
        )[:-1]

        # RR
        self.df_parti_rr = load_participant_data(
            participant_number=self.parti_no,
            signal_names=RR_SIGNALS,
            bandpassed=True,
            normalized=False,
            segmented=False,
            class_name="RR",
            load_labels=False,
            time_offset=True,
        )

        parti_data, parti_labels = load_participant_data(
            participant_number=self.parti_no,
            signal_names=RR_SIGNALS,
            bandpassed=True,
            normalized=True,
            segmented=True,
            class_name="RR",
            load_labels=True,
            time_offset=True,
        )
        self.df_parti_segmented_rr = put_metadata_to_segments(
            segments=parti_data, labels=parti_labels, parti_no=self.parti_no
        )[:-1]

        # RR - Reference:
        self.df_ref_rr = load_participant_data(
            participant_number=self.parti_no,
            signal_names=["resp_ref"],
            segmented=False,
            time_offset=True,
            class_name="RR",
        )

        self.df_ref_segmented_rr = load_participant_data(
            participant_number=self.parti_no,
            signal_names=["resp_ref"],
            segmented=True,
            time_offset=True,
            class_name="RR",
        )[:-1]

    @property
    def signal_type(self) -> str:
        if self.current_signal_name.startswith("ecg"):
            return "ECG"
        elif self.current_signal_name.startswith("ppg"):
            return "PPG"
        elif self.current_signal_name.startswith("scg"):
            return "SCG"
        elif self.current_signal_name.startswith("mi"):
            return "MI"

        raise ValueError("Unknown signal type!")
