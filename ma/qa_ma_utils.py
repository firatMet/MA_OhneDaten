from typing import Literal

import numpy as np
import pandas as pd
from itertools import combinations

from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from ma.utils import load_all_signals
from sklearn.model_selection import train_test_split

ALL_PARTI = list(range(1, 21))


def select_test_dataset(
    label_distributions: dict[str, float],
    desired_distribution: dict[str, float],
    set_len: int = 4,
) -> dict[str, list[int]]:
    def calculate_proportions(selected_sets, *label_types):
        totals = {
            label: sum(label_distributions[set_id][label] for set_id in selected_sets)
            for label in label_types
        }
        total_sum = sum(totals.values())

        return {label: totals[label] / total_sum for label in label_types}

    # Generalized function to calculate the score for how close the proportions are to the target
    def score_proportions(selected_sets, label_types, target_proportions):
        proportions = calculate_proportions(selected_sets, *label_types)
        score = np.sqrt(
            sum(
                (proportions[label] - target_proportions[label]) ** 2
                for label in label_types
            )
        )
        return score

    # Find the best combination of sets for any number of labels
    def find_best_sets(label_types, target_proportions):
        best_sets = None
        best_score = float("inf")

        for sets in combinations(label_distributions.keys(), set_len):
            current_score = score_proportions(sets, label_types, target_proportions)
            if current_score < best_score:
                best_score = current_score
                best_sets = sets

        best_sets = list(map(str, best_sets))
        return best_sets, best_score

    # Usage for 'sq' proportions (good, bad, noisy)
    sq_label_types = ["good", "bad", "noisy"]
    desired_sq_proportions = {
        "good": desired_distribution["good_percentage"],
        "bad": desired_distribution["bad_percentage"],
        "noisy": desired_distribution["noisy_percentage"],
    }
    best_sets_sq, _ = find_best_sets(sq_label_types, desired_sq_proportions)

    # Usage for 'ma' proportions (ma, no_ma)
    ma_label_types = ["ma", "no_ma"]
    desired_ma_proportions = {
        "ma": desired_distribution["ma_percentage"],
        "no_ma": desired_distribution["no_ma_percentage"],
    }
    best_sets_ma, _ = find_best_sets(ma_label_types, desired_ma_proportions)

    return {"sq": list(best_sets_sq), "ma": list(best_sets_ma)}


def calculate_distribution_of_labels_from_sqi_df(
    sqi_df: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float]]:
    parti_distributions = {}
    total_counts = {"good": 0, "bad": 0, "noisy": 0, "ma": 0, "no_ma": 0}

    for parti_no in range(1, 21):
        df_filtered_parti = sqi_df[sqi_df["parti"] == str(parti_no)]
        counts = {
            "good": len(df_filtered_parti[df_filtered_parti["sq"] == 0]),
            "bad": len(df_filtered_parti[df_filtered_parti["sq"] == 1]),
            "noisy": len(df_filtered_parti[df_filtered_parti["sq"] == 2]),
            "ma": len(df_filtered_parti[df_filtered_parti["ma"] == 1]),
            "no_ma": len(df_filtered_parti[df_filtered_parti["ma"] == 0]),
        }
        parti_distributions[parti_no] = counts

        # Add to total counts
        for key in counts:
            total_counts[key] += counts[key]

    total_sq = total_counts["good"] + total_counts["bad"] + total_counts["noisy"]
    total_ma = total_counts["ma"] + total_counts["no_ma"]

    percentages = {
        "good_percentage": total_counts["good"] / total_sq,
        "bad_percentage": total_counts["bad"] / total_sq,
        "noisy_percentage": total_counts["noisy"] / total_sq,
        "ma_percentage": total_counts["ma"] / total_ma,
        "no_ma_percentage": total_counts["no_ma"] / total_ma,
    }

    return parti_distributions, percentages


def calculate_distribution_of_labels_from_data(
    df_segmented: list[pd.DataFrame],
) -> tuple[dict[str, float], dict[str, float]]:
    parti_distributions = {
        parti_no: {"good": 0, "bad": 0, "noisy": 0, "ma": 0, "no_ma": 0}
        for parti_no in range(1, 21)
    }
    for segment in df_segmented:
        parti_no = int(segment.attrs["signal"].split("parti: ")[-1].split(" |")[0])
        sq_labels = [key for key in segment.attrs.keys() if "sq" in key]
        ma_labels = [key for key in segment.attrs.keys() if "ma" in key]
        for sq_label in sq_labels:
            match segment.attrs[sq_label]:
                case 0:
                    parti_distributions[parti_no]["good"] += 1
                case 1:
                    parti_distributions[parti_no]["bad"] += 1
                case 2:
                    parti_distributions[parti_no]["noisy"] += 1

        for ma_label in ma_labels:
            match segment.attrs[ma_label]:
                case 0:
                    parti_distributions[parti_no]["no_ma"] += 1
                case 1:
                    parti_distributions[parti_no]["ma"] += 1

    total_counts = {"good": 0, "bad": 0, "noisy": 0, "ma": 0, "no_ma": 0}
    for distribution in parti_distributions.values():
        for key in total_counts:
            total_counts[key] += distribution[key]

    total_sq = total_counts["good"] + total_counts["bad"] + total_counts["noisy"]
    total_ma = total_counts["ma"] + total_counts["no_ma"]

    percentages = {
        "good_percentage": total_counts["good"] / total_sq,
        "bad_percentage": total_counts["bad"] / total_sq,
        "noisy_percentage": total_counts["noisy"] / total_sq,
        "ma_percentage": total_counts["ma"] / total_ma,
        "no_ma_percentage": total_counts["no_ma"] / total_ma,
    }

    return parti_distributions, percentages


def calculate_accuracies_from_cm(
    cm: np.ndarray, label: Literal["ma", "sq"]
) -> pd.DataFrame:
    assert label in ["ma", "sq"]

    match label:
        case "sq":
            good_row = cm[0, :]
            bad_row = cm[1, :]
            noisy_row = cm[2, :]

            good_acc = good_row[0] / np.sum(good_row)
            bad_acc = bad_row[1] / np.sum(bad_row)
            noisy_acc = noisy_row[2] / np.sum(noisy_row)

            total_acc = (cm[0, 0] + cm[1, 1] + cm[2, 2]) / (np.sum(cm))

            return pd.DataFrame(
                {
                    "good_acc": [good_acc],
                    "bad_acc": [bad_acc],
                    "noisy_acc": [noisy_acc],
                    "total_acc": [total_acc],
                }
            )
        case "ma":
            no_ma_raw = cm[0, :]
            ma_row = cm[1, :]

            no_ma_acc = no_ma_raw[0] / np.sum(no_ma_raw)
            ma_acc = ma_row[1] / np.sum(ma_row)

            total_acc = (cm[0, 0] + cm[1, 1]) / (np.sum(cm))

            return pd.DataFrame(
                {"no_ma_acc": [no_ma_acc], "ma_acc": [ma_acc], "total_acc": [total_acc]}
            )


def load_train_test_datasets(
    signal_type: Literal["ECG", "PPG", "SCG", "MI"],
    label: Literal["ma", "sq"],
    class_name: Literal["HR", "RR"] = "HR",
    shuffle_train_data: bool = False,
) -> dict[str, pd.DataFrame]:
    assert class_name in ["HR", "RR"], "Please give a valid class name, HR or RR"
    assert label in ["ma", "sq"]

    X_train = []
    y_test = []
    X_validation = []
    y_validation = []
    X_test = []
    y_train = []

    all_data = load_all_signals(
        signal_type=signal_type, bandpassed=True, normalized=True, class_name=class_name
    )
    parti_distributions, desired_distributions = (
        calculate_distribution_of_labels_from_data(df_segmented=all_data)
    )
    test_parti = select_test_dataset(
        label_distributions=parti_distributions,
        desired_distribution=desired_distributions,
        set_len=4,
    )[label]
    test_parti = [int(parti) for parti in test_parti]

    remaining_distributions = {
        key: value
        for key, value in parti_distributions.items()
        if key not in test_parti
    }
    validation_parti = select_test_dataset(
        label_distributions=remaining_distributions,
        desired_distribution=desired_distributions,
        set_len=1,
    )[label]
    validation_parti = [int(parti) for parti in validation_parti]

    train_parti = np.setdiff1d(
        np.arange(1, 21), np.concatenate((test_parti, validation_parti))
    )

    print(
        f"Train Parti: {train_parti}, Validation Parti: {validation_parti}, Test Parti: {test_parti}"
    )

    for segment in all_data:
        for col in segment.columns:
            if segment.attrs[f"{col}_{label}"] == -1:
                continue
            # Remove Noisy Signals if MA
            if label == "ma" and segment.attrs[f"{col}_sq"] == 2:
                continue
            if (
                int(segment.attrs[f"signal"].split("parti: ")[-1].split(" |")[0])
                in test_parti
            ):
                X_test.append(segment[col].values)
                y_test.append(segment.attrs[f"{col}_{label}"])
            elif (
                int(segment.attrs[f"signal"].split("parti: ")[-1].split(" |")[0])
                in validation_parti
            ):
                X_validation.append(segment[col].values)
                y_validation.append(segment.attrs[f"{col}_{label}"])
            elif (
                int(segment.attrs[f"signal"].split("parti: ")[-1].split(" |")[0])
                in train_parti
            ):
                X_train.append(segment[col].values)
                y_train.append(segment.attrs[f"{col}_{label}"])

    # Convert the data to np array:
    X_train = np.array(X_train)
    X_validation = np.array(X_validation)
    X_test = np.array(X_test)

    if shuffle_train_data:
        X_train, y_train = shuffle(X_train, y_train, random_state=42)

    # Convert labels to categorical format:
    match label:
        case "sq":
            num_classes = 3
        case "ma":
            num_classes = 2
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_validation = to_categorical(y_validation, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "X_test": X_test,
        "y_test": y_test,
    }
