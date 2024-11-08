import pandas as pd
import numpy as np
from ma.gp_utils import (
    load_deep_model,
    load_sqi_model,
)
from ma.signal_recoverer import SignalRecoverer
import pathlib
import os
from ma.qa_ma_sqi import predict_from_sqi_model
from ma.qa_ma_networks import predict_from_network


MODEL_PATH = pathlib.Path("ma/models/").resolve()


class GPPerfectClassifier(SignalRecoverer):
    def __init__(
        self, parti_no: int, signal_names: list[str] = None, class_name: str = "HR"
    ) -> None:
        super().__init__(parti_no, signal_names, class_name)

        self._calculate_hr_and_rr()
        self._allocate_results()


class GPModelClassifier(SignalRecoverer):
    def __init__(
        self, parti_no: int, signal_names: list[str] = None, class_name: str = "HR"
    ) -> None:
        super().__init__(parti_no, signal_names, class_name)

        self._load_classifiers()
        self._predict_labels()
        self._calculate_hr_and_rr()
        self._allocate_results()

    def _load_classifiers(self) -> None:
        self.ma_model = {}
        self.sq_model = {}
        all_files = os.listdir(path=MODEL_PATH)
        ma_filenames = self.signal_type + "_ma_" + self.class_name
        sq_filenames = self.signal_type + "_sq_" + self.class_name

        ma_classifier = [
            classifier
            for classifier in all_files
            if classifier.startswith(ma_filenames)
        ]
        sq_classifier = [
            classifier
            for classifier in all_files
            if classifier.startswith(sq_filenames)
        ]

        if any("weights" in element for element in ma_classifier):  # Deep Network
            # Load model architecture
            self.ma_model["model"] = load_deep_model(model_files=ma_classifier)
            self.ma_model["type"] = "network"
        elif any("features" in element for element in ma_classifier):  # SV ML Model
            self.ma_model["model"], self.ma_model["features"] = load_sqi_model(
                model_files=ma_classifier
            )
            self.ma_model["type"] = "sqi_model"

        if any("weights" in element for element in sq_classifier):  # Deep Network
            self.sq_model["model"] = load_deep_model(model_files=sq_classifier)
            self.sq_model["type"] = "network"
        elif any("features" in element for element in sq_classifier):  # SV ML Model
            self.sq_model["model"], self.sq_model["features"] = load_sqi_model(
                model_files=sq_classifier
            )
            self.sq_model["type"] = "sqi_model"

    def _predict_labels(self) -> None:
        self.sq_labels = {}
        self.ma_labels = {}
        parti_no = (
            self.df_parti_segmented[0]
            .attrs["signal"]
            .split("parti: ")[-1]
            .split(" ")[0]
        )
        for signal_name in self.df_parti_segmented[0].columns:
            match self.sq_model["type"]:
                case "sqi_model":
                    self.sq_labels[signal_name] = predict_from_sqi_model(
                        signal_name,
                        parti_no,
                        self.class_name,
                        self.sq_model["model"],
                        self.sq_model["features"],
                    )
                case "network":
                    self.sq_labels[signal_name] = predict_from_network(
                        df_segmented=self.df_parti_segmented,
                        classifier=self.sq_model["model"],
                        signal_name=signal_name,
                    )

            match self.ma_model["type"]:
                case "sqi_model":
                    self.ma_labels[signal_name] = predict_from_sqi_model(
                        signal_name,
                        parti_no,
                        self.class_name,
                        self.ma_model["model"],
                        self.ma_model["features"],
                    )
                case "network":
                    self.ma_labels[signal_name] = predict_from_network(
                        df_segmented=self.df_parti_segmented,
                        classifier=self.ma_model["model"],
                        signal_name=signal_name,
                    )

        # Update the labels in the segments:
        for signal_name in self.df_parti_segmented[0].columns:
            for segment, sq_label, ma_label in zip(
                self.df_parti_segmented,
                self.sq_labels[signal_name],
                self.ma_labels[signal_name],
            ):
                segment.attrs[f"{signal_name}_sq"] = sq_label
                segment.attrs[f"{signal_name}_ma"] = ma_label
