import numpy as np
import pandas as pd
import pickle
import os

from ma.qa_ma_sqi import load_sqi_df, SupervisedLearning, SQIS

from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from imblearn.ensemble import RUSBoostClassifier
from sklearn.pipeline import Pipeline


def get_features(label: str, signal: str, class_name: str) -> list[str]:
    match class_name:
        case "HR":
            match signal:
                case "ECG":
                    match label:
                        case "sq":
                            return SQIS
                        case "ma":
                            return [
                                "skew",
                                "f80",
                                "zc_mean",
                                "mean_dist_of_peaks",
                                "peak_freq",
                                "mean_dist_maf_peaks",
                                "std_dist_maf_peaks",
                            ]
                case "PPG":
                    match label:
                        case "sq":
                            return [
                                "zc_mean",
                                "zc_derivative",
                                "skew",
                                "zc",
                                "cluster_number_maf",
                                "f80",
                                "peak_freq",
                                "kurt",
                            ]
                        case "ma":
                            return [
                                "skew",
                                "cluster_number_maf",
                                "zc_mean",
                                "kurt",
                                "f80",
                                "zc",
                                "iqr",
                                "zc_derivative",
                                "mean_dist_maf_peaks",
                                "std_dist_maf_peaks",
                                "peak_freq",
                            ]
                case "SCG":
                    match label:
                        case "sq":
                            return None
                        case "ma":
                            return None
                case "MI":
                    match label:
                        case "sq":
                            return None
                        case "ma":
                            return None

        case "RR":
            match signal:
                case "ECG":
                    match label:
                        case "sq":
                            return None
                        case "ma":
                            return None
                case "PPG":
                    match label:
                        case "sq":
                            return None
                        case "ma":
                            return None
                case "SCG":
                    match label:
                        case "sq":
                            return None
                        case "ma":
                            return None
                case "MI":
                    match label:
                        case "sq":
                            return None
                        case "ma":
                            return None


def get_train_dataset(signal: str, label: str, features: list) -> tuple:
    sqi_df = load_sqi_df(signal=signal, label=label)
    sL = SupervisedLearning(sqi_df=sqi_df, label=label, features=features)
    return sL._X_train, sL._y_train


def search_hyper_parameters(
    model, distributions, X_train, y_train, rusboost: bool = False
) -> pd.DataFrame:
    cv = KFold(n_splits=10, shuffle=True)

    if not rusboost:
        rcv = RandomizedSearchCV(model, distributions, cv=cv, n_jobs=-1, n_iter=1000)
    else:
        distributions = {
            f"rusboost__estimator__{key}": value for key, value in distributions.items()
        }
        distributions["rusboost__n_estimators"] = [10, 20, 30, 40, 50, 60]
        distributions["rusboost__learning_rate"] = [0.01, 0.1, 10, 100]
        print(distributions)

        rusboost = RUSBoostClassifier(estimator=model)
        pipe = Pipeline([("rusboost", rusboost)])
        rcv = RandomizedSearchCV(pipe, distributions, cv=cv, n_jobs=-1, n_iter=1000)

    rcv.fit(X_train, y_train)

    return pd.DataFrame(rcv.cv_results_).sort_values(
        by="mean_test_score", ascending=False
    )


def save_hp_df(
    df_hp: pd.DataFrame, model_name: str, signal: str, label: str, class_name: str
) -> None:
    # Ensure the directory exists
    os.makedirs("tmp", exist_ok=True)
    with open(f"tmp/hp_{model_name}_{signal}_{label}_{class_name}.pkl", "wb") as file:
        pickle.dump(df_hp, file)


def load_hp_df(
    model_name: str, signal: str, label: str, class_name: str
) -> pd.DataFrame:
    with open(f"tmp/hp_{model_name}_{signal}_{label}_{class_name}.pkl", "rb") as file:
        df_hp = pickle.load(file)
    return df_hp
