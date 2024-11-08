# pip install imbalanced-learn scikit-learn

import pandas as pd
import numpy as np
import pickle
from copy import deepcopy
import os

from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from ma.qa_ma_utils import (
    calculate_accuracies_from_cm,
    calculate_distribution_of_labels_from_sqi_df,
    select_test_dataset,
)
from typing import Literal
import joblib
import re
from sklearn.utils import shuffle


from ma.utils import fft_filter, wavelet_transform
from ma.model_mapping import MODEL_MAPPING_HR, MODEL_MAPPING_RR

SQIS = [
    "kurt",
    "skew",
    "mean",
    "std",
    "peak",
    "rms",
    "mavfd",
    "mavsd",
    "zc",
    "zc_mean",
    "zc_derivative",
    "iqr",
    "mean_dist_of_peaks",
    "mean_dist_maf_peaks",
    "std_dist_maf_peaks",
    "cluster_number_maf",
    "peak_amp",
    "peak_freq",
    "mean_amp",
    "med_amp",
    "f80",
    "wavelet_energy",
    "wavelet_entropy",
    "wavelet_std",
    "wavelet_mean",
]

###########################
### Supervised Learning ###
###########################


class SupervisedLearning:
    def __init__(
        self,
        sqi_df: pd.DataFrame,
        label: Literal["ma", "sq"],
        class_name: Literal["HR", "RR"],
        standardize: bool = True,
        features: list[str] = None,
        k_fold: int = None,
    ) -> None:
        # Data Preperation:
        assert label in ["ma", "sq"]

        self.label = label
        self.features = features
        self.class_name = class_name
        self.signal_type = re.sub(r"\d+", "", sqi_df["signal"].iloc[0])[:3]

        self._sqi_df = sqi_df
        self.parti_distributions, self.label_distributions = (
            calculate_distribution_of_labels_from_sqi_df(sqi_df=sqi_df)
        )
        self._prepare_data(features=features, label=label)

        # Data Preprocesing:
        if standardize:
            self._standardize_features()

        # Model Installation:
        self.models = {}
        self._init_models()

        # Cross Validation:
        if not k_fold:
            k_fold = 20 - len(self.test_parti)
        self.k_fold = k_fold

    def predict(
        self,
        x: np.ndarray,
        ground_truth: np.ndarray = None,
        model_names: list[str] = None,
    ) -> dict:
        if not model_names:
            model_names = [model_name for model_name in self.models.keys()]
        results = {model_name: {"y_pred": [], "cm": []} for model_name in model_names}
        for model_name in model_names:
            y_pred = self.models[model_name].predict(x)
            results[model_name]["y_pred"] = y_pred
            if ground_truth is not None:
                cm = confusion_matrix(y_pred=y_pred, y_true=ground_truth)
                results[model_name]["cm"] = cm
                results[model_name]["accs"] = calculate_accuracies_from_cm(
                    cm, label=self.label
                )

        return results

    def evaluate_models(self, cross_validation: bool = True) -> None:
        match cross_validation:
            case True:
                self._evaluate_models_w_cv()
            case False:
                self._evaluate_models_wo_cv()

    def _evaluate_models_wo_cv(self) -> None:
        # Evaluation WITHOUT Cross Validation
        for model_name, model in self.models.items():
            # Train:
            print(f"\n---------Start to train w/o CV: {model_name}---------\n")
            self._train_model(model=model)

            # Prediction & Evaluation
            y_pred_train = model.predict(self._X_train)
            y_pred_test = model.predict(self._X_test)
            cm_train = confusion_matrix(self._y_train, y_pred_train)
            cm_test = confusion_matrix(self._y_test, y_pred_test)

            print(
                f"""---- {model_name} ----
                  \n Train:\n {cm_train}
                  \n {calculate_accuracies_from_cm(cm_train, label = self.label)}
                  \n Test:\n {cm_test}
                  \n {calculate_accuracies_from_cm(cm_test, label = self.label)}"""
            )

    def _evaluate_models_w_cv(self) -> None:
        # Evaluation WITH Cross Validation
        for model_name, model in self.models.items():
            match self.label:
                case "sq":
                    validation_results = {
                        "good_acc": [],
                        "bad_acc": [],
                        "noisy_acc": [],
                        "total_acc": [],
                    }
                case "ma":
                    validation_results = {
                        "no_ma_acc": [],
                        "ma_acc": [],
                        "total_acc": [],
                    }
            # Train:
            print(f"\n---------Start to train with CV: {model_name}---------\n")
            X_train = deepcopy(self._X_train)
            y_train = deepcopy(self._y_train)
            parti_order = deepcopy(self._parti_order)
            self._train_model(model=model)

            # Evaluation
            group_kfold = GroupKFold(n_splits=self.k_fold)

            # Initialize fold counter
            fold_idx = 1

            # Manually iterate over each fold split to print participants
            for train_idx, val_idx in group_kfold.split(
                X_train, y_train, groups=parti_order
            ):
                print(f"\n--- Fold {fold_idx} ---")

                # Get the participant IDs in the training and validation sets
                train_participants = parti_order[train_idx]
                val_participants = parti_order[val_idx]

                # Print unique participants in the train and validation sets
                print(f"Training Participants: {np.unique(train_participants)}")
                print(f"Validation Participants: {np.unique(val_participants)}")

                # Fit with the current fold
                model.fit(X_train[train_idx], y_train[train_idx])

                y_pred_validation = model.predict(X_train[val_idx])
                cm_validation = confusion_matrix(y_train[val_idx], y_pred_validation)

                validation_result = calculate_accuracies_from_cm(
                    cm_validation, label=self.label
                )

                print(
                    f"""\n Validation:\n {cm_validation}
                    \n {validation_result}"""
                )

                for col in validation_result.columns:
                    validation_results[col].append(validation_result[col][0])

                fold_idx += 1
            averaged_validation_results = {
                key: np.mean(value) for key, value in validation_results.items()
            }
            print(
                f"The Averaged Validation Results of the Model : {model_name} is: {averaged_validation_results}"
            )

    def _train_model(self, model) -> None:
        X_train = deepcopy(self._X_train)
        y_train = deepcopy(self._y_train)
        model.fit(X_train, y_train)

    def _init_models(self) -> None:
        match self.class_name:
            case "HR":
                self.models = MODEL_MAPPING_HR.get(self.label, {}).get(self.signal_type)
            case "RR":
                self.models = MODEL_MAPPING_RR.get(self.label, {}).get(self.signal_type)

    @staticmethod
    def logarithmic_kernel(X, Y):
        return np.log(np.dot(X, Y.T))

    def _standardize_features(self):
        scaler = StandardScaler()
        self._X_train = scaler.fit_transform(self._X_train)
        self._X_test = scaler.fit_transform(self._X_test)

    def _prepare_data(
        self,
        label: Literal["ma", "sq"],
        features: list[str] = None,
    ):
        # Clean out the unknown labels (-1):
        self._sqi_df = self._sqi_df[self._sqi_df[label] != -1]

        if features:
            self._sqi_df = self._sqi_df[features + [label, "parti"]]
        else:
            features = SQIS

        self.test_parti = select_test_dataset(
            label_distributions=self.parti_distributions,
            desired_distribution=self.label_distributions,
        )[self.label]

        print(f"The entire dataset has the distribution: {self.label_distributions}")
        print(f"The test dataset contains participants: {self.test_parti}")

        sqi_train = self._sqi_df[~self._sqi_df["parti"].isin(self.test_parti)]
        sqi_test = self._sqi_df[self._sqi_df["parti"].isin(self.test_parti)]

        self._X_train = sqi_train[features].values
        self._y_train = sqi_train[label].values
        self._X_test = sqi_test[features].values
        self._y_test = sqi_test[label].values
        self._parti_order = sqi_train["parti"].values

        # Shuffle the training data
        self._X_train, self._y_train, self._parti_order = shuffle(
            self._X_train, self._y_train, self._parti_order, random_state=42
        )

    def save_model(self, model_name: str, file_name: str) -> None:
        # Save the network architecture to JSON format
        model = self.models[model_name]
        joblib.dump(model, f"{file_name}.pkl")
        joblib.dump(self.features, f"{file_name}_features.pkl")


########################
### SQI Calculations ###
########################


def calculate_sqis(data: list[pd.DataFrame]) -> pd.DataFrame:
    sqis = {
        f"{sqi}": [] for sqi in (SQIS + ["sq", "ma", "parti", "signal", "segment_no"])
    }
    for segment in data:
        # Timeseries:
        sqis["kurt"].extend(kurtosis(segment.values))
        sqis["skew"].extend(skew(segment.values))
        sqis["mean"].extend(np.mean(segment.values, axis=0))
        sqis["std"].extend(np.std(segment.values, axis=0))
        sqis["peak"].extend(np.max(abs(segment.values), axis=0))
        sqis["rms"].extend(np.sqrt(np.mean(np.square(segment.values), axis=0)))
        sqis["mavfd"].extend(_calculate_mavfd(segment))
        sqis["mavsd"].extend(_calculate_mavsd(segment))
        sqis["zc"].extend(_calculate_zero_crossings_first_at_zero(segment))
        sqis["zc_mean"].extend(_calculate_zero_crossings_mean_at_zero(segment))
        sqis["zc_derivative"].extend(_calculate_zero_crossings_derivative(segment))
        sqis["iqr"].extend(
            segment.apply(lambda x: x.quantile(0.75) - x.quantile(0.25)).tolist()
        )
        sqis["mean_dist_of_peaks"].extend(_calculate_mean_distance_of_peaks(segment))
        sqis["mean_dist_maf_peaks"].extend(_calculate_dist_maf_peaks(segment))
        sqis["std_dist_maf_peaks"].extend(_calculate_std_dist_maf_peaks(segment))
        sqis["cluster_number_maf"].extend(_calculate_cluster_number_maf(segment))

        # FFT:
        segment_fft = fft_filter(df_segment=segment)

        sqis["peak_amp"].extend(np.max(segment_fft.values, axis=0))
        sqis["peak_freq"].extend(np.argmax(segment_fft.values, axis=0))
        sqis["mean_amp"].extend(np.mean(segment_fft.values, axis=0))
        sqis["med_amp"].extend(np.median(segment_fft.values, axis=0))
        sqis["f80"].extend(_calculate_f80(segment_fft))

        # Wavelet Transform
        segment_wavelet = wavelet_transform(df_segment=segment)

        sqis["wavelet_energy"].extend(_calculate_wavelet_energy(segment_wavelet))
        sqis["wavelet_entropy"].extend(_calculate_wavelet_entropy(segment_wavelet))
        sqis["wavelet_std"].extend(_calculate_wavelet_std(segment_wavelet))
        sqis["wavelet_mean"].extend(_calculate_wavelet_mean(segment_wavelet))

        # Label
        sqis[f"sq"].extend(
            [value for key, value in segment.attrs.items() if key.endswith("sq")]
        )
        sqis[f"ma"].extend(
            [value for key, value in segment.attrs.items() if key.endswith("ma")]
        )

        # Meta:
        sqis["parti"].extend(
            [segment.attrs[f"signal"].split("parti: ")[-1].split(" |")[0]]
            * len(segment.columns)
        )
        sqis["signal"].extend(segment.columns)
        sqis["segment_no"].extend(
            [int((segment.attrs[f"signal"].split("segment: ")[-1]))]
            * len(segment.columns)
        )

    return pd.DataFrame(sqis)


def predict_from_sqi_model(
    signal_name: str, parti_no: str, class_name: str, classifier, features: list[str]
) -> None:
    # Load the sqi dataframe:
    signal_type = re.sub(r"\d+", "", signal_name).upper()
    with open(f"feature_data/sqi_df_{signal_type}_{class_name}.pkl", "rb") as file:
        sqi_df = pickle.load(file)

    # Find the features values of the corresponding segment:
    filtered_sqi_df = sqi_df[
        (sqi_df["parti"] == parti_no) & (sqi_df["signal"] == signal_name)
    ]
    X = filtered_sqi_df[features].values

    # Prediction
    prediction = classifier.predict(X)

    return prediction


def save_sqi_df(
    sqi_df: pd.DataFrame,
    signal: Literal["ecg", "ppg", "scg", "mu"],
    class_name: Literal["HR", "RR"],
) -> None:
    # Ensure the directory exists
    os.makedirs("tmp", exist_ok=True)

    with open(f"feature_data/sqi_df_{signal}_{class_name}.pkl", "wb") as file:
        pickle.dump(sqi_df, file)
    return


def load_sqi_df(
    signal: Literal["ecg", "ppg", "scg", "mu"],
    label: Literal["sq", "ma"],
    class_name: Literal["HR", "RR"],
) -> pd.DataFrame:
    with open(f"feature_data/sqi_df_{signal}_{class_name}.pkl", "rb") as file:
        sqi_df = pickle.load(file)

    # Remove the noisy signals in ma detection
    if label == "ma":
        sqi_df = sqi_df[sqi_df["sq"] != 2]

    return sqi_df


def _calculate_mavfd(segment: pd.DataFrame) -> list[float]:
    absolute_first_differences = abs(segment.diff(periods=-1).iloc[:-1])
    mavfd = np.mean(absolute_first_differences, axis=0)
    return mavfd


def _calculate_mavsd(segment: pd.DataFrame) -> list[float]:
    # Compute the differences (x_i - x_{i+2})
    absolute_second_differences = abs(segment.diff(periods=-2).iloc[:-1])
    mavsd = np.mean(absolute_second_differences, axis=0)
    return mavsd


def _calculate_zero_crossings_mean_at_zero(segment: pd.DataFrame) -> list[int]:
    # Move the signal st its mean value sits in the 0-center.
    mean_segment = np.mean(segment, axis=0)
    segment = segment - mean_segment

    # Zero-crossings:
    signs = np.sign(segment)
    zc = (abs(np.diff(signs, axis=0)) > 1).sum(axis=0)
    return zc


def _calculate_zero_crossings_first_at_zero(segment: pd.DataFrame) -> list[int]:
    # Move the signal zo the 0-center.
    zc_segment = segment.copy()
    zc_segment -= zc_segment.iloc[0]

    # Zero-crossings:
    signs = np.sign(zc_segment)
    zc = (abs(np.diff(signs, axis=0)) > 1).sum(axis=0)
    return zc


def _calculate_zero_crossings_derivative(segment: pd.DataFrame) -> list[int]:
    # Take the derivative of the signals:
    zc_segment = segment.diff()
    cols = zc_segment.columns
    num_cols = len(cols)

    # Initialize zero-crossing counters for each column:
    zcs = [0] * num_cols

    # Initialize the old sign values for each column (starting from row 1):
    old_signs = [np.sign(zc_segment[col].iloc[1]) for col in cols]

    # Iterate over rows of the DataFrame starting from row 2
    for row_no, row in enumerate(zc_segment.iterrows()):
        if row_no == 0:
            continue

        for i, col in enumerate(cols):
            current_sign = np.sign(row[1][col])

            # Check for a zero-crossing (sign change) for each column
            if current_sign != old_signs[i] and current_sign != 0:
                zcs[i] += 1

            # Update the old sign only if the current sign is non-zero
            if current_sign != 0:
                old_signs[i] = current_sign

    return zcs


def _calculate_mean_distance_of_peaks(segment: pd.DataFrame) -> float:
    """
    Mean value of the distance of the peakss to each other that exceeds the threshold value (3 * std)

    Parameters
    ----------
    segment : pd.DataFrame

    Returns
    -------
    float

    """
    mean_distance_of_peaks = []
    for col in segment.columns:
        three_std = np.std(segment[col].values) * 3
        peak_indexes = np.where(segment[col].values > three_std)
        if len(peak_indexes[0]) == 0 or len(peak_indexes[0]) == 1:
            mean_distance_of_peaks.append(0)
            continue
        peak_times = (
            (segment.index[peak_indexes] - segment.index[peak_indexes][0])
            .total_seconds()
            .astype(float)
        )
        diff_peak_times = peak_times.diff()
        diff_peak_times = diff_peak_times.dropna()
        mean_distance_of_peaks.append(np.mean(diff_peak_times))
    return mean_distance_of_peaks


def _calculate_dist_maf_peaks(segment: pd.DataFrame) -> float:
    """
    Mean value of the points that are detected by the moving average filter from the second derivative of the segments

    Parameters
    ----------
    segment : pd.DataFrame

    Returns
    -------
    float
    """
    mean_distance_of_maf_peaks = []
    for col in segment.columns:
        df = segment[col]
        df_first_derivative = df.diff()
        df_second_derivative = df_first_derivative.diff()
        threshold = np.std(df_second_derivative) * 3 + np.mean(df_second_derivative)
        peaks = _moving_average_filter(
            arr=df_second_derivative.values, window_size=10, threshold=threshold
        )
        mean_distance_of_peaks = np.mean(np.diff(peaks))
        mean_distance_of_maf_peaks.append(mean_distance_of_peaks)
    return mean_distance_of_maf_peaks


def _calculate_std_dist_maf_peaks(segment: pd.DataFrame) -> float:
    """
    STD value of the points that are detected by the moving average filter from the second derivative of the segments

    Parameters
    ----------
    segment : pd.DataFrame

    Returns
    -------
    float
    """
    std_distance_of_peaks = []
    for col in segment.columns:
        df = segment[col]
        df_first_derivative = df.diff()
        df_second_derivative = df_first_derivative.diff()
        threshold = np.std(df_second_derivative) * 2
        peaks = _moving_average_filter(
            arr=df_second_derivative.values, window_size=10, threshold=threshold
        )
        if len(peaks) == 0:
            std_distance_of_peaks.append(0)
            continue
        std_distance_of_peak = np.std(np.diff(peaks))
        std_distance_of_peaks.append(std_distance_of_peak)
    return std_distance_of_peaks


def _calculate_cluster_number_maf(segment: pd.DataFrame) -> float:
    clusters = []
    for col in segment.columns:
        df = segment[col]
        df_first_derivative = df.diff()
        df_second_derivative = df_first_derivative.diff()
        threshold = np.std(df_second_derivative) * 2
        peaks = _moving_average_filter(
            arr=df_second_derivative.values, window_size=10, threshold=threshold
        )
        if len(peaks) == 0:
            return [0, 0, 0, 0]
        differences = np.diff(peaks)
        cluster = np.sum(differences > 5) + 1
        clusters.append(cluster)
    return clusters


def _calculate_f80(segment_fft: pd.DataFrame) -> list[float]:
    power_80 = np.sum(segment_fft.values, axis=0) * 0.8
    power_below_80 = np.cumsum(segment_fft.values, axis=0) < power_80

    idxs = np.zeros(len(segment_fft.columns), dtype=int)
    # Loop through each column index (0, 1, 2, 3)
    for col in range(len(segment_fft.columns)):
        indices = np.where(power_below_80[:, col])
        if indices[0].size == 0:
            idxs[col] = 0
            continue
        idxs[col] = indices[0][-1]

    last_freqs_below_80 = segment_fft.index[idxs]
    return last_freqs_below_80


def _moving_average_filter(
    arr: np.ndarray, window_size: int, threshold: float
) -> list[float]:

    padded_end = np.full(window_size, arr[-1])
    padded_begin = np.full(window_size, arr[0])
    arr = np.concatenate((padded_begin, arr, padded_end))
    filtered_data = np.zeros(len(arr))

    for i in range(len(arr) - window_size + 1):
        filtered_data[i : i + window_size] = np.std(arr[i : i + window_size]) * 2
    filtered_data = np.abs(arr - filtered_data)
    filtered_data = filtered_data[window_size:-window_size]
    filtered_idx = np.where(filtered_data > threshold)

    if len(filtered_idx[0]) < 2:
        return [0, 0]

    return filtered_idx[0]


def _calculate_wavelet_energy(segment_wavelet: pd.DataFrame) -> pd.Series:
    """Calculate the energy of wavelet coefficients."""
    energies = np.sum(np.square(segment_wavelet), axis=0)
    return energies


def _calculate_wavelet_entropy(segment_wavelet: pd.DataFrame) -> pd.Series:
    """Calculate the entropy of wavelet coefficients."""

    def shannon_entropy(coeffs):
        """Calculate Shannon entropy."""
        if len(coeffs) == 0:
            return 0
        prob, _ = np.histogram(coeffs, bins=10, density=True)
        prob = prob[prob > 0]  # Remove zero probabilities
        return -np.sum(prob * np.log(prob))

    entropies = segment_wavelet.apply(lambda col: shannon_entropy(col))
    return entropies


def _calculate_wavelet_std(segment_wavelet: pd.DataFrame) -> pd.Series:
    """Calculate the standard deviation of wavelet coefficients."""
    std_devs = segment_wavelet.std()
    return std_devs


def _calculate_wavelet_mean(segment_wavelet: pd.DataFrame) -> pd.Series:
    """Calculate the mean of wavelet coefficients."""
    means = segment_wavelet.mean()
    return means
