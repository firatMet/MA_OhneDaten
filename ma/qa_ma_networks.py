import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Dense, Flatten, SimpleRNN, LSTM
from tensorflow.keras.models import Sequential
from ma.qa_ma_utils import calculate_accuracies_from_cm
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from typing import Literal
import os


def predict_from_network(
    df_segmented: list[pd.DataFrame], classifier, signal_name: str
) -> list[int]:
    X_batch = np.stack(
        [np.reshape(segment[signal_name].values, (1, 256)) for segment in df_segmented]
    )
    X_batch = X_batch.reshape(-1, 256)
    # Make predictions for all segments at once
    predictions_batch = classifier.predict(X_batch)

    # Get the argmax of each prediction to get the predicted labels
    predictions = np.argmax(predictions_batch, axis=1)

    return predictions


class LongShortTermMemory:
    def __init__(
        self,
        datasets: dict[str, pd.DataFrame],
        label: Literal["ma", "sq"],
        depth: int = 2,
        activations: list[str] = ["relu"],
        regularization: bool = False,
    ) -> None:
        assert isinstance(
            datasets, dict
        ), """Please give valid datasets.
        The correct format is: {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}"""
        assert label in ["ma", "sq"]

        self.label = label

        self._prepare_data(datasets=datasets)

        self._depth = depth
        self._activations = activations
        self._initialize_network(regularization=regularization)
        self._compile_network()

    def train_network(
        self,
        epochs: int = 50,
        batch_size: int = 32,
        callback: bool = True,
        thresholds: dict[str, float] = None
    ) -> None:
        if callback:
            os.makedirs("tensorboard", exist_ok=True)
            tensorboard = TensorBoard(log_dir=f"tensorboard/lstm_{self.label}")
            callbacks = [
                MetricsCallback(
                    self.X_train,
                    self.y_train,
                    self.X_validation,
                    self.y_validation,
                    label=self.label,
                    thresholds=thresholds
                ),
                tensorboard,
            ]
        else:
            callbacks = None
        self._network.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0,
            validation_data=(self.X_validation, self.y_validation),
            callbacks=callbacks,
        )

    def predict(self, X: np.ndarray, ground_truth: np.ndarray = None) -> list[int]:
        results = {}

        y_pred = self._network.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
        results["y_pred"] = y_pred

        if ground_truth is not None:
            # If the y_pred is the output of softmax:
            if np.shape(ground_truth)[1] > 1:
                ground_truth = np.argmax(ground_truth, axis=1)
            cm = confusion_matrix(y_pred=y_pred, y_true=ground_truth)
            results["cm"] = cm
            results["accs"] = calculate_accuracies_from_cm(cm, label=self.label)

        return results

    def evaluate_network(self) -> dict[str, float]:
        loss, accuracy = self._network.evaluate(self.X_test, self.y_test)
        return {"test_loss": loss, "test_accuracy": accuracy}

    def save_network(self, name: str) -> None:
        # Save the network architecture to JSON format
        network_json = self._network.to_json()
        with open(f"{name}.json", "w") as json_file:
            json_file.write(network_json)

        # Save the weights
        self._network.save_weights(f"{name}.weights.h5")

    def _initialize_network(self, regularization: bool = False) -> None:
        self._network = Sequential()

        if not regularization:
            kernel_regularizer = None
        else:
            kernel_regularizer = l2(0.01)

        if len(self._activations) != self._depth:
            if len(self._activations) > 1:
                raise NotImplementedError(
                    "Please either define all the activations, or define only one activation!"
                )
            self._activations = [self._activations[0]] * self._depth

        # Input Layer (Simple RNN):
        # First parameter: number of neurons in the the input layer.
        # Input shape: to arrange each neuron to the data.
        self._network.add(
            LSTM(256, input_shape=(self.X_train.shape[1], 1), activation="relu")
        )
        # Hidden Layers:
        for activation in self._activations:
            self._network.add(
                Dense(64, activation=activation, kernel_regularizer=kernel_regularizer)
            )
        # Output Layer:
        match self.label:
            case "sq":
                output_number = 3
            case "ma":
                output_number = 2
        self._network.add(Dense(output_number, activation="softmax"))

    def _compile_network(self) -> None:
        # Optimization algorithm: Adam : Adaptive learning rate optimization algorithm.
        self._network.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

    def _prepare_data(self, datasets: dict[str, list[float]]) -> None:
        X_train = datasets["X_train"]
        X_validation = datasets["X_validation"]
        X_test = datasets["X_test"]

        self._y_train = datasets["y_train"]
        self._y_validation = datasets["y_validation"]
        self._y_test = datasets["y_test"]
        # scaler = StandardScaler()
        scaler = StandardScaler()
        self._X_train = scaler.fit_transform(X_train.reshape(-1, 256)).reshape(
            -1, 256, 1
        )
        self._X_validation = scaler.fit_transform(
            X_validation.reshape(-1, 256)
        ).reshape(-1, 256, 1)
        self._X_test = scaler.transform(X_test.reshape(-1, 256)).reshape(-1, 256, 1)

    @property
    def X_train(self) -> None:
        return self._X_train

    @property
    def y_train(self) -> None:
        return self._y_train

    @property
    def X_validation(self) -> None:
        return self._X_validation

    @property
    def y_validation(self) -> None:
        return self._y_validation

    @property
    def X_test(self) -> None:
        return self._X_test

    @property
    def y_test(self) -> None:
        return self._y_test


class RNN:
    def __init__(
        self,
        datasets: dict[str, pd.DataFrame],
        label: Literal["ma", "sq"],
        depth: int = 2,
        activations: list[str] = ["relu"],
        regularization: bool = False,
    ) -> None:
        assert isinstance(
            datasets, dict
        ), """Please give valid datasets.
        The correct format is: {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}"""
        assert label in ["ma", "sq"]

        self.label = label

        self._prepare_data(datasets=datasets)

        self._depth = depth
        self._activations = activations
        self._initialize_network(regularization=regularization)
        self._compile_network()

    def train_network(
        self,
        epochs: int = 50,
        batch_size: int = 32,
        callback: bool = True,
        thresholds: dict[str, float] = None
    ) -> None:
        if callback:
            os.makedirs("tensorboard", exist_ok=True)
            tensorboard = TensorBoard(log_dir=f"tensorboard/rnn_{self.label}")
            callbacks = [
                MetricsCallback(
                    self.X_train,
                    self.y_train,
                    self.X_validation,
                    self.y_validation,
                    label=self.label,
                    thresholds=thresholds
                ),
                tensorboard,
            ]
        else:
            callbacks = None
        self._network.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0,
            validation_data=(self.X_validation, self.y_validation),
            callbacks=callbacks,
        )

    def predict(self, X: np.ndarray, ground_truth: np.ndarray = None) -> list[int]:
        results = {}

        y_pred = self._network.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
        results["y_pred"] = y_pred

        if ground_truth is not None:
            # If the y_pred is the output of softmax:
            if np.shape(ground_truth)[1] > 1:
                ground_truth = np.argmax(ground_truth, axis=1)
            cm = confusion_matrix(y_pred=y_pred, y_true=ground_truth)
            results["cm"] = cm
            results["accs"] = calculate_accuracies_from_cm(cm, label=self.label)

        return results

    def evaluate_network(self) -> dict[str, float]:
        loss, accuracy = self._network.evaluate(self.X_test, self.y_test)
        return {"test_loss": loss, "test_accuracy": accuracy}

    def save_network(self, name: str) -> None:
        # Save the network architecture to JSON format
        network_json = self._network.to_json()
        with open(f"{name}.json", "w") as json_file:
            json_file.write(network_json)
        # Save the weights
        self._network.save_weights(f"{name}.weights.h5")

    def _initialize_network(self, regularization: bool = False) -> None:
        self._network = Sequential()
        if not regularization:
            kernel_regularizer = None
        else:
            kernel_regularizer = l2(0.01)

        if len(self._activations) != self._depth:
            if len(self._activations) > 1:
                raise NotImplementedError(
                    "Please either define all the activations, or define only one activation!"
                )
            self._activations = [self._activations[0]] * self._depth

        # Input Layer (Simple RNN):
        # First parameter: number of neurons in the the input layer.
        # Input shape: to arrange each neuron to the data.
        self._network.add(
            SimpleRNN(
                256,
                input_shape=(self.X_train.shape[1], 1),
                activation="relu",
                kernel_regularizer=kernel_regularizer,
            )
        )
        # Hidden Layers:
        for activation in self._activations:
            self._network.add(Dense(64, activation=activation))
        # Output Layer:
        match self.label:
            case "sq":
                output_number = 3
            case "ma":
                output_number = 2
        self._network.add(Dense(output_number, activation="softmax"))

    def _compile_network(self) -> None:
        # Optimization algorithm: Adam : Adaptive learning rate optimization algorithm.
        self._network.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

    def _prepare_data(self, datasets: dict[str, list[float]]) -> None:
        X_train = datasets["X_train"]
        X_test = datasets["X_test"]
        X_validation = datasets["X_validation"]

        self._y_train = datasets["y_train"]
        self._y_validation = datasets["y_validation"]
        self._y_test = datasets["y_test"]
        # scaler = StandardScaler()
        # Reshape X_train to (samples, 256, 1)
        self._X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        # Reshape X_validation to (samples, 256, 1)
        self._X_validation = X_validation.reshape(
            (X_validation.shape[0], X_validation.shape[1], 1)
        )
        # Reshape X_test to (samples, 256, 1)
        self._X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    @property
    def X_train(self) -> None:
        return self._X_train

    @property
    def y_train(self) -> None:
        return self._y_train

    @property
    def X_validation(self) -> None:
        return self._X_validation

    @property
    def y_validation(self) -> None:
        return self._y_validation

    @property
    def X_test(self) -> None:
        return self._X_test

    @property
    def y_test(self) -> None:
        return self._y_test


class CNN:
    def __init__(
        self,
        datasets: dict[str, pd.DataFrame],
        label: Literal["ma", "sq"],
        depth: int = 5,
        filters: list[int] = [64, 128, 256, 512, 512],
        activation: str = "relu",
        padding: str = "same",
        kernel_size: int = 3,
        pool_size: int = 2,
        stride: int = 2,
        regularization: bool = False,
    ) -> None:
        # General Network Structute is taken from VGGNets
        # CONV -> CONV -> POOL ->  ... ->  CONV -> CONV -> POOL -> Flatten -> ReLU -> DropOut -> ReLu -> DropOut -> SoftMax
        # CONV -> CONV -> POOL : Convolution Blocks
        # Flatten -> ReLU -> DropOut -> ReLu -> DropOut : Fully Connected Layers
        # SoftMax: Output Layer
        assert depth == len(
            filters
        ), f"Depth ({depth}) and the length of filters ({len(filters)}) must match!"
        assert label in ["ma", "sq"]

        
        self.label = label
        match self.label:
            case "sq":
                self._num_classes = 3
            case "ma":
                self._num_classes = 2
        self._filters = filters
        self._activation = activation
        self._padding = padding
        self._kernel_size = kernel_size
        self._pool_size = pool_size
        self._stride = stride

        # Prepare Datasets
        self.label = label
        self._prepare_data(datasets=datasets)

        # Empty Network:
        self.network = models.Sequential()
        self._input_shape = np.shape(self.X_train)[1:]

        # Add Convolutional Blocks:
        for d in range(depth):
            self._add_conv_block(depth=d, regularization=regularization)

        # Add Fully Connected Layers:
        self._add_fully_connected_layers()

        # Compile the Model:
        self.network.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.network.summary()

    def predict(self, X: np.ndarray, ground_truth: np.ndarray = None) -> dict:
        results = {}

        y_pred = self.network.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
        results["y_pred"] = y_pred

        if ground_truth is not None:

            cm = confusion_matrix(y_pred=y_pred, y_true=ground_truth)
            results["cm"] = cm
            results["accs"] = calculate_accuracies_from_cm(cm, label=self.label)

        return results

    def evaluate_network(self) -> None:
        test_loss, test_acc = self.network.evaluate(self.X_test, self.y_test)
        return (test_loss, test_acc)

    def train_network(
        self,
        epochs: int = 50,
        batch_size: int = 32,
        callback: bool = True,
        thresholds: dict[str, float] = None
    ) -> dict:  # ?
        if callback:
            os.makedirs("tensorboard", exist_ok=True)
            tensorboard = TensorBoard(log_dir=f"tensorboard/cnn_{self.label}")
            callbacks = [
                MetricsCallback(
                    self.X_train,
                    self.y_train,
                    self.X_validation,
                    self.y_validation,
                    label=self.label,
                    thresholds=thresholds
                ),
                tensorboard,
            ]
        else:
            callbacks = None

        history = self.network.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0,
            validation_data=(self.X_validation, self.y_validation),
            callbacks=callbacks,
        )

        return history

    def save_network(self, name: str) -> None:
        # Save the network architecture to JSON format
        network_json = self.network.to_json()
        with open(f"{name}.json", "w") as json_file:
            json_file.write(network_json)

        # Save the weights
        self.network.save_weights(f"{name}.weights.h5")

    def _add_fully_connected_layers(self) -> None:
        # Fully Connected Layers
        self.network.add(layers.Flatten())
        self.network.add(layers.Dense(units=4096, activation="relu"))
        self.network.add(layers.Dropout(0.5))
        self.network.add(layers.Dense(units=4096, activation="relu"))
        self.network.add(layers.Dropout(0.5))
        self.network.add(layers.Dense(units=self._num_classes, activation="softmax"))

    def _add_conv_block(self, depth: int, regularization: bool = False) -> None:
        # If it's not the first layer, you don't have to provide any input shape.
        input_shape = None
        if not regularization:
            kernel_regularizer = None
        else:
            kernel_regularizer = l2(0.01)
        if depth == 0:
            input_shape = self._input_shape

        self.network.add(
            layers.Conv1D(
                filters=self._filters[depth],
                kernel_size=self._kernel_size,
                padding=self._padding,
                activation=self._activation,
                input_shape=input_shape,
            )
        )

        input_shape = None

        self.network.add(
            layers.Conv1D(
                filters=self._filters[depth],
                kernel_size=self._kernel_size,
                padding=self._padding,
                activation=self._activation,
                input_shape=input_shape,
                kernel_regularizer=kernel_regularizer,
            )
        )

        self.network.add(layers.MaxPooling1D(pool_size=2, strides=2))

    def _prepare_data(self, datasets: dict[str, list[float]]) -> None:
        X_train = datasets["X_train"]
        X_validation = datasets["X_validation"]
        y_train = datasets["y_train"]
        
        
        X_test = datasets["X_test"]
        y_validation = datasets["y_validation"]
        y_test = datasets["y_test"]
        # Reshape X_train to (samples, 256, 1)
        self._X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        # Convert y_train to class labels
        self._y_train = np.argmax(y_train, axis=1)
        # Reshape X_validation to (samples, 256, 1)
        self._X_validation = X_validation.reshape((X_validation.shape[0], X_validation.shape[1], 1))
        # Convert y_validation to class labels
        self._y_validation = np.argmax(y_validation, axis=1)
        # Reshape X_test to (samples, 256, 1)
        self._X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        # Convert y_test to class labels
        self._y_test = np.argmax(y_test, axis=1)

    @property
    def X_train(self) -> None:
        return self._X_train

    @property
    def y_train(self) -> None:
        return self._y_train
    
    @property
    def X_validation(self) -> None:
        return self._X_validation

    @property
    def y_validation(self) -> None:
        return self._y_validation

    @property
    def X_test(self) -> None:
        return self._X_test

    @property
    def y_test(self) -> None:
        return self._y_test


class MLP:
    def __init__(
        self,
        datasets: dict[str, pd.DataFrame],
        label: Literal["ma", "sq"],
        depth: int = 2,
        activations: list[str] = ["relu"],
        learning_rate: float = 0.001,
    ) -> None:
        assert isinstance(
            datasets, dict
        ), """Please give valid datasets.
        The correct format is: {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}"""
        assert label in ["ma", "sq"]

        self.label = label

        self._X_train = datasets["X_train"]
        self._y_train = datasets["y_train"]
        self._X_validation = datasets["X_validation"]
        self._y_validation = datasets["y_validation"]
        self._X_test = datasets["X_test"]
        self._y_test = datasets["y_test"]

        self._depth = depth
        self._activations = activations
        self._initialize_network()
        self._compile_network(lr=learning_rate)

    def train_network(
        self,
        epochs: int = 50,
        batch_size: int = 32,
        callback: bool = True,
        thresholds: dict[str, float] = None
    ) -> None:
        if callback:
            os.makedirs("tensorboard", exist_ok=True)
            tensorboard = TensorBoard(log_dir=f"tensorboard/mlp_{self.label}")
            callbacks = [
                MetricsCallback(
                    self.X_train,
                    self.y_train,
                    self.X_validation,
                    self.y_validation,
                    label=self.label,
                    thresholds=thresholds
                ),
                tensorboard,
            ]
        else:
            callbacks = None
        self._network.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0,
            validation_data=(self.X_validation, self.y_validation),
            callbacks=callbacks,
        )

    def predict(self, X: np.ndarray, ground_truth: np.ndarray = None) -> list[int]:
        results = {}

        y_pred = self._network.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
        results["y_pred"] = y_pred

        if ground_truth is not None:
            # If the y_pred is the output of softmax:
            if np.shape(ground_truth)[1] > 1:
                ground_truth = np.argmax(ground_truth, axis=1)
            cm = confusion_matrix(y_pred=y_pred, y_true=ground_truth)
            results["cm"] = cm
            results["accs"] = calculate_accuracies_from_cm(cm, label=self.label)

        return results

    def evaluate_network(self) -> dict[str, float]:
        loss, accuracy = self._network.evaluate(self.X_test, self.y_test)
        return {"test_loss": loss, "test_accuracy": accuracy}

    def _initialize_network(self) -> None:
        self._network = Sequential()

        if len(self._activations) != self._depth:
            if len(self._activations) > 1:
                raise NotImplementedError(
                    "Please either define all the activations, or define only one activation!"
                )
            self._activations = [self._activations[0]] * self._depth

        # Input Layer (Dense: Fully Connected Layer):
        # First parameter: number of neurons in the the input layer.
        # Input shape: to arrange each neuron to the data.
        self._network.add(
            Dense(256, activation="relu", input_shape=(self.X_train.shape[1],))
        )
        # Hidden Layers:
        for activation in self._activations:
            self._network.add(Dense(64, activation=activation))
        # Output Layer:
        match self.label:
            case "sq":
                output_number = 3
            case "ma":
                output_number = 2
        self._network.add(Dense(output_number, activation="softmax"))

    def _compile_network(self, lr: float) -> None:
        # Optimization algorithm: Adam : Adaptive learning rate optimization algorithm.
        optimizer = Adam(learning_rate=lr)
        self._network.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )

    def save_network(self, name: str) -> None:
        # Save the network architecture to JSON format
        network_json = self._network.to_json()
        with open(f"{name}.json", "w") as json_file:
            json_file.write(network_json)
        # Save the weights
        self._network.save_weights(f"{name}.weights.h5")

    @property
    def X_train(self) -> None:
        return self._X_train

    @property
    def y_train(self) -> None:
        return self._y_train

    @property
    def X_validation(self) -> None:
        return self._X_validation

    @property
    def y_validation(self) -> None:
        return self._y_validation

    @property
    def X_test(self) -> None:
        return self._X_test

    @property
    def y_test(self) -> None:
        return self._y_test


class MetricsCallback(tf.keras.callbacks.Callback):
    """
    This class serves to configure the logs after each epoch in model training !
    """

    def __init__(self, x_train, y_train, x_validation, y_validation, label, thresholds : dict[str, float] = None):
        super(MetricsCallback, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_validation = x_validation
        self.y_validation = y_validation
        self.label = label

        # Early Stopping
        self.thresholds = thresholds

    def on_epoch_end(self, epoch, logs=None):
        y_pred_train = np.argmax(self.model.predict(self.x_train), axis=1)
        y_pred_validation = np.argmax(self.model.predict(self.x_validation), axis=1)
        if self.y_validation.ndim > 1:
            y_train = np.argmax(self.y_train, axis=1)
            y_validation = np.argmax(self.y_validation, axis=1)
        else:
            y_train = self.y_train
            y_validation = self.y_validation
        cm_train = confusion_matrix(y_train, y_pred_train)
        cm_validation = confusion_matrix(y_validation, y_pred_validation)
        # print(cm)
        accs_train = calculate_accuracies_from_cm(cm_train, label=self.label)
        accs_validation = calculate_accuracies_from_cm(cm_validation, label=self.label)
        print("--------- Accuracies on the Train Dataset: --------- \n")
        print(accs_train)
        print("\n--------- Accuracies on the Validation Dataset: --------- \n")
        print(accs_validation)

        # EARLY STOP
        if self.thresholds:
            stop = False
            for threshold_key, threshold_value in self.thresholds.items():
                if accs_validation[threshold_key][0] > threshold_value:
                    stop = True
                else:
                    stop = False
                    break
            if stop:
                print(f"\nEARLY STOP as validation ma_acc exceeded!\n")
                self.model.stop_training = True  # Stop training if the threshold is exceeded

        # tn, fp, fn, tp = cm.ravel()

        # print(f"Epoch {epoch + 1}:")
        # print(f"True Positives: {tp}")
        # print(f"True Negatives: {tn}")
        # print(f"False Positives: {fp}")
        # print(f"False Negatives: {fn}")
