import pandas as pd
import numpy as np

from itertools import groupby, count
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
import GPy


peak_detection_func_mapping = {
    "ECG": ecg_r_peak_detection,
    "PPG": ppg_peak_detection,
    "SCG": scg_peak_detection,
    "MI": mi_peak_detection,
}
peak_detection_ref_func_mapping = {
    "ECG": ecg_r_peak_detection,
    "PPG": ecg_r_peak_detection,
    "SCG": scg_peak_detection,
    "MI": scg_peak_detection,
}


class SignalRecoverer(ABC):
    def __init__(
        self, parti_no: int, signal_names: list[str] = None, class_name: str = "HR"
    ) -> None:
        self.parti_no = parti_no
        self.signal_names = signal_names
        self.class_name = class_name
        self.ma_indexes = {}
        self.ref_signal = "ecg_ref" if self.class_name == "HR" else "resp_ref"

        self.peaks_parti = {}
        self.df_parti_hr_rr = {}

        self.peaks_ref = {}
        self.df_ref_hr_rr = {}

        self._initialize_participant_data()
        self._initialize_models()

        self._train_data_found = False
        # self._calculate_hr_and_rr()
        # self._allocate_results()

    def _initialize_participant_data(self) -> None:
        # Not-Segmented data:
        self.df_parti = load_participant_data(
            participant_number=self.parti_no,
            signal_names=self.signal_names,
            bandpassed=True,
            time_offset=True,
            class_name=self.class_name,
        )

        # Segmented data
        parti_data, parti_labels = load_participant_data(
            participant_number=self.parti_no,
            signal_names=self.signal_names,
            bandpassed=True,
            normalized=False,
            segmented=True,
            class_name=self.class_name,
            load_labels=True,
            time_offset=True,
        )
        self.df_parti_segmented = put_metadata_to_segments(
            segments=parti_data, labels=parti_labels, parti_no=self.parti_no
        )[:-1]

        # Reference Data
        self.df_ref = load_participant_data(
            participant_number=self.parti_no,
            signal_names=[self.ref_signal],
            time_offset=True,
            class_name=self.class_name,
        )

        self.df_ref_segmented = load_participant_data(
            participant_number=self.parti_no,
            signal_names=[self.ref_signal],
            segmented=True,
            time_offset=True,
            class_name=self.class_name,
        )[:-1]

        # First signal name
        self.current_signal_name = self.df_parti_segmented[0].columns[0]

    def _initialize_models(self) -> None:
        self.models = {}

        rbf_kernel = GPy.kern.RBF(input_dim=1, variance=10.0, lengthscale=5.0)
        periodic_kernel_respiration = GPy.kern.PeriodicExponential(input_dim=1, variance=10.0, lengthscale=1.0, period=1.0)
        periodic_kernel_mayer = GPy.kern.PeriodicExponential(input_dim=1, variance=10.0, lengthscale=1.0, period=10.0)

        # Combine kernels
        combined_kernel = rbf_kernel + periodic_kernel_respiration + periodic_kernel_mayer
        data_placeholder = np.zeros((1, 1))
        self.models["gpy"] = GPy.models.GPRegression(data_placeholder, data_placeholder, kernel=combined_kernel, normalizer=True)

    def _calculate_hr_and_rr(self):
        # Need to select the correct peak detection algorithm based on the class & signal names
        func_peak_detection = peak_detection_func_mapping.get(self.signal_type)
        func_peak_detection_ref = peak_detection_ref_func_mapping.get(self.signal_type)
        match self.class_name:
            case "HR":
                # Ref signal: cECG
                self.peaks_ref = func_peak_detection_ref(
                    df_segmented=self.df_ref_segmented,
                    signal_name=self.ref_signal,
                    height_threshold=0.349,
                )
            case "RR":
                # Ref Signal: Resp Ref
                self.peaks_ref = func_peak_detection_ref(
                    df_segmented=self.df_ref_segmented,
                    signal_name="resp_ref",
                    distance=20,
                    dynamic_height_multiplier=0.1,
                    prominence=0.1,
                )
        self.df_ref_hr_rr = calculate_hr_or_rr(
            df_parti=self.df_ref,
            peaks=self.peaks_ref,
            class_name=self.class_name,
        )

        for signal_name in self.signal_names:
            # Get the R-Peaks:
            self.peaks_parti[signal_name] = func_peak_detection(
                df_segmented=self.df_parti_segmented,
                signal_name=signal_name,
            )

            # Get the HR-Signals:
            self.df_parti_hr_rr[signal_name] = calculate_hr_or_rr(
                df_parti=self.df_parti,
                peaks=self.peaks_parti[signal_name],
                class_name=self.class_name,
            )

    def _allocate_results(self) -> None:
        self.recovered_signals = {}
        for signal_name in self.signal_names:
            self.recovered_signals[signal_name] = {}
            for model_name in self.models.keys():
                self.recovered_signals[signal_name][model_name] = {}
                self.recovered_signals[signal_name][model_name]["recovered_signal"] = (
                    self.df_parti_hr_rr[signal_name].copy()
                )
                self.recovered_signals[signal_name][model_name]["old_x"] = []
                self.recovered_signals[signal_name][model_name]["old_y"] = []
                self.recovered_signals[signal_name][model_name]["new_x"] = []
                self.recovered_signals[signal_name][model_name]["predicted_stds"] = []
                self.recovered_signals[signal_name][model_name]["train_signals"] = []
                self.recovered_signals[signal_name][model_name]["ma_errors"] = []
                self.recovered_signals[signal_name][model_name]["overall_errors"] = []

    def recover_signals(self) -> None:
        # Main loop: Iterate through the segments with motion artifacts & recover the signals !
        self._get_ma_indexes()
        for signal_name in self.signal_names:
            self.current_signal_name = signal_name
            for model_name, model in self.models.items():
                for ma_index in self.ma_indexes[self.current_signal_name]:
                    train_data, test_data = self._get_train_test_data(
                        ma_index=ma_index,
                        model_name=model_name,
                    )
                    self._train_models(
                        train_data=train_data,
                        model_name=model_name,
                        model=model,
                    )
                    if self._train_data_found:
                        self._recover_signal(
                            test_data=test_data,
                            model_name=model_name,
                            model=model,
                        )

    def _get_ma_indexes(self) -> None:
        for signal_name in self.signal_names:
            df = self.df_parti_hr_rr[signal_name]
            ma_indexes = df[df['ma_labels'] == 1].index
            ma_ilocs = df.index.get_indexer(ma_indexes)
            ma_ilocs = [list(group) for _, group in groupby(ma_ilocs, key=lambda x, c=count(): x - next(c))]
            self.ma_indexes[signal_name] = [[df.index[group[0] -1],df.index[group[-1]] +1 ] for group in ma_ilocs]

    def _train_models(
        self,
        train_data: dict[str, np.ndarray],
        model_name: str,
        model,
    ) -> None:
        # No good segments near the motion artifact:
        if len(train_data["X_train"]) == 0:
            self._train_data_found = False
            return

        # self.scaler_X = StandardScaler()
        # self.scaler_y = StandardScaler()

        # X_train_scaled = self.scaler_X.fit_transform(train_data["X_train"])
        # y_train_scaled = self.scaler_y.fit_transform(
        #     train_data["y_train"].reshape(-1, 1)
        # ).ravel()

        model.set_XY(train_data["X_train"], train_data["y_train"])
        model.optimize()
        self.recovered_signals[self.current_signal_name][model_name][
            "train_signals"
        ].append(train_data)
        self._train_data_found = True

    def _recover_signal(
        self, test_data: dict[str, np.ndarray], model_name: str, model
    ) -> None:
        # Predict the y values on the reference signal's timestamps!
        x_start = np.argmin(abs(self.df_ref_hr_rr.index - test_data["X_ma"][0]))
        x_end = np.argmin(abs(self.df_ref_hr_rr.index - test_data["X_ma"][-1])) + 1
        x = np.array(self.df_ref_hr_rr.index[x_start:x_end]).reshape(-1, 1)
        y_pred, sigma = model.predict(x)
        y_pred = y_pred.flatten()
        sigma = sigma.flatten()
        # # Save the predicted values into a copy of the self.df_parti_hr (and remove the ones with motion artifacts)!:
        df_model = self.recovered_signals[self.current_signal_name][model_name][
            "recovered_signal"
        ]
        # Remove the motion artifact portion:
        indices = [x[0] for x in test_data["X_ma"]]
        df_model = df_model.drop(index=df_model.loc[indices[0] : indices[-1]].index)
        # Tuck the predicted portion in:
        labels = {f"{self.current_signal_name}_sq": 0, "signal": "RECOVERED"}
        column_name = df_model.columns[0]
        predicted_portion = pd.DataFrame(
            {column_name: y_pred, "labels": [labels] * len(y_pred)}, index=x.flatten()
        )
        df_model = pd.concat(
            [
                df_model.loc[: indices[0]],
                predicted_portion,
                df_model.loc[indices[-1] :],
            ],
        ).sort_index()
        self.recovered_signals[self.current_signal_name][model_name][
            "recovered_signal"
        ] = df_model
        # Save other data:
        self.recovered_signals[self.current_signal_name][model_name]["old_x"].append(
            test_data["X_ma"].flatten()
        )
        self.recovered_signals[self.current_signal_name][model_name]["old_y"].append(
            test_data["y_ma"].flatten()
        )
        self.recovered_signals[self.current_signal_name][model_name]["new_x"].append(
            x.flatten()
        )
        self.recovered_signals[self.current_signal_name][model_name][
            "predicted_stds"
        ].append(sigma)

    def _get_train_test_data(
        self,
        ma_index: list[int, int],
        model_name: str,
        segment_length: int = 10,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        df_ma = self.df_parti_hr_rr[self.current_signal_name].loc[
            ma_index[0] : ma_index[-1]
        ]
        df_recovered = self.recovered_signals[self.current_signal_name][model_name][
            "recovered_signal"
        ]
        df_train_left = self._get_df_train_left(
            df_recovered=df_recovered,
            df_ma=df_ma,
            segment_length=segment_length,
        )
        df_train_right = self._get_df_train_right(
            df_recovered=df_recovered,
            df_ma=df_ma,
            ma_index=ma_index,
            segment_length=segment_length,
        )

        df_train = pd.concat([df_train_left, df_train_right])
        df_train = df_train[df_train["hr"] > 45]
        df_train = df_train[df_train["hr"] < 75]

        X_train = np.reshape(df_train.index, (-1, 1))
        y_train = np.reshape(df_train[self.class_name.lower()].values, (-1, 1))
        X_ma = np.reshape(df_ma.index, (-1, 1))
        y_ma = np.reshape(df_ma[self.class_name.lower()].values, (-1, 1))

        return {"X_train": X_train, "y_train": y_train}, {"X_ma": X_ma, "y_ma": y_ma}

    def _get_df_train_right(
        self,
        df_recovered: pd.DataFrame,
        df_ma: pd.DataFrame,
        ma_index: list[int, int],
        segment_length: int,
    ) -> pd.DataFrame:

        df_raw = self.df_parti_hr_rr[self.current_signal_name]
        stop_idx = np.argmin(
            np.abs(
                df_recovered.index
                - min(
                    df_ma.index[-1] + segment_length,
                    len(self.df_parti_hr_rr[self.current_signal_name]),
                )
            )
        )

        df_train_right = df_recovered.iloc[
            df_recovered.index.get_loc(df_ma.index[-1]) + 1 : stop_idx
        ]
        bad_noisy_quality_idx = []
        for index, row in df_train_right.iterrows():
            if row["sq_labels"] != 0 or row["ma_labels"] != 0:
                bad_noisy_quality_idx.append(index)

        df_train_right = df_train_right.drop(bad_noisy_quality_idx)

        return df_train_right

    def _get_df_train_left(
        self,
        df_recovered: pd.DataFrame,
        df_ma: pd.DataFrame,
        segment_length: int,
    ) -> pd.DataFrame:
        start_idx = np.argmin(
            np.abs(df_recovered.index - max(df_ma.index[0] - segment_length, 0))
        )
        start_idx = df_recovered.index[start_idx]
        if df_ma.index[0] == 0:  # Extreme Case: MA at start
            start_idx = len(df_ma) + 1

        df_train_left = df_recovered.loc[start_idx : df_ma.index[0]-0.001]
        bad_noisy_quality_idx = []
        for index, row in df_train_left.iterrows():
            if row["sq_labels"] != 0:
                bad_noisy_quality_idx.append(index)

        df_train_left = df_train_left.drop(bad_noisy_quality_idx)

        return df_train_left

    def calculate_ma_error(self) -> None:
        for signal_name in self.signal_names:
            for model_name, value in self.recovered_signals[signal_name].items():
                df_recovered = value["recovered_signal"]
                new_x = value["new_x"]
                errors_recovered = calculate_hr_error_ma_intervals(
                    df_parti_hr=df_recovered,
                    df_ref_hr=self.df_ref_hr_rr,
                    ma_indexes=new_x,
                    recovered=True,
                )
                errors_raw = calculate_hr_error_ma_intervals(
                    df_parti_hr=self.df_parti_hr_rr[signal_name],
                    df_ref_hr=self.df_ref_hr_rr,
                    ma_indexes=self.ma_indexes[signal_name],
                    recovered=False,
                )

                self.recovered_signals[signal_name][model_name]["ma_errors"] = {
                    "errors_raw": errors_raw,
                    "errors_recovered": errors_recovered,
                }

    def calculate_overall_error(self) -> None:
        for signal_name in self.signal_names:
            for model_name, value in self.recovered_signals[signal_name].items():
                errors_recovered = calculate_hr_error_overall(
                    df_parti_hr=value["recovered_signal"],
                    df_ref_hr=self.df_ref_hr_rr,
                )

                errors_raw = calculate_hr_error_overall(
                    df_parti_hr=self.df_parti_hr_rr[signal_name],
                    df_ref_hr=self.df_ref_hr_rr,
                )

                self.recovered_signals[signal_name][model_name]["overall_errors"] = {
                    "errors_raw": errors_raw,
                    "errors_recovered": errors_recovered,
                }

    def get_ma_errors(self, signal_name: str, model_name: str) -> pd.DataFrame:
        return pd.DataFrame(
            self.recovered_signals[signal_name][model_name]["ma_errors"]
        )

    def get_overall_errors(self, signal_name: str, model_name: str) -> pd.DataFrame:
        return pd.DataFrame(
            self.recovered_signals[signal_name][model_name]["overall_errors"]
        )

    def plot_recovered_signals(
        self, model_name: str, signal_name: str, mark_train_signals: bool = False
    ) -> Figure:
        # Plot the whole signal & plot the +-1.95 std values fill between on the predicted values where you recovered.
        # Use the 1x2 plotting function is gp_visu & plot this as the 3rd row!
        # Maybe also put the error rates to the titles
        fig = plt.figure(figsize=(20, 9))
        axes = fig.subplots(nrows=3, ncols=1, sharex=True)

        df_parti_segmented = [df[[signal_name]] for df in self.df_parti_segmented]

        hr_rr = self.df_parti_hr_rr[signal_name].columns[0]
        label_prefix = ""
        match hr_rr:
            case "hr":
                label_prefix = "Heart Rate"
                bpm_label = "Heart Rate [bpm]"
            case "rr":
                label_prefix = "Respiratory Rate"
                bpm_label = "Respiratory Rate [bpm]"

        fig = plot_signals_with_hr(
            df_parti_segmented=df_parti_segmented,
            df_parti_hr=self.df_parti_hr_rr[signal_name],
            df_ref=self.df_ref,
            df_ref_hr=self.df_ref_hr_rr,
            df_parti=self.df_parti,
            peaks_parti=self.peaks_parti[signal_name],
            peaks_ref=self.peaks_ref,
            class_name=self.class_name,
            fig=fig,
            axes=axes,
        )

        # Shade the MA areas in HR plots:
        ax2 = fig.get_axes()[1]
        for ma_index in self.ma_indexes[signal_name]:
            ax2.axvspan(
                self.df_parti_hr_rr[signal_name]
                .loc[ma_index[0] : ma_index[-1]]
                .index[0],
                self.df_parti_hr_rr[signal_name]
                .loc[ma_index[0] : ma_index[-1]]
                .index[-1],
                color="lightgray",
                alpha=0.4,
            )
        ax2.set_xlabel("")
        # ax2.set_ylabel(bpm_label)

        # Plot the 3rd row:
        df = self.recovered_signals[signal_name][model_name]["recovered_signal"]
        # There is a bug in the algorithm, sometimes gets duplicates:
        df = df[~df.index.duplicated()]

        stds = self.recovered_signals[signal_name][model_name]["predicted_stds"]
        new_x = self.recovered_signals[signal_name][model_name]["new_x"]
        train_signals = self.recovered_signals[signal_name][model_name]["train_signals"]
        ax3 = fig.get_axes()[2]

        ax3.plot(
            df.index,
            df[self.class_name.lower()],
            color="Green",
            marker="x",
            label=f"{label_prefix} - Recovered Signal",
        )
        for recovered_interval in new_x:
            ax3.axvspan(
                recovered_interval[0],
                recovered_interval[-1],
                color="lightgray",
                alpha=0.4,
            )
        # Add stds to the plot:
        for x, std in zip(new_x, stds):
            y_pred = df.loc[x, self.class_name.lower()].values
            ax3.plot(x, y_pred + 1.96 * std, linestyle="--", color="blue", alpha=0.5)
            ax3.plot(x, y_pred - 1.96 * std, linestyle="--", color="blue", alpha=0.5)
            # ax3.errorbar(
            #     x,
            #     y_pred,
            #     yerr=1.96 * std,
            #     fmt='o',
            #     color="blue",
            #     alpha=0.5,
            # )

        ax3.plot(
            self.df_ref_hr_rr.index,
            self.df_ref_hr_rr[self.class_name.lower()],
            color="orange",
            alpha=0.3,
            marker="o",
            label=f"{label_prefix} - Reference Signal",
        )

        ax3.plot(
            [],
            [],
            linestyle="--",
            color="blue",
            label="95% confidence interval",
        )

        if mark_train_signals:
            for train_signal in train_signals:
                x = train_signal["X_train"].flatten()
                y = train_signal["y_train"].flatten()
                ax3.scatter(x, y, color="purple", alpha=1, marker="o")

            ax3.scatter([], [], color="purple", label="Train Signals")

        ax3.legend()
        ax3.grid()
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel(bpm_label)
        fig.suptitle(
            f"Participant No: {self.parti_no} | Signal: {signal_name} | GP Kernel: {model_name}"
        )

        return fig

    @property
    def model_names(self) -> list[str]:
        return list(self.models.keys())

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
