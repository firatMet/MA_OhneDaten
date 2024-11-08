# Open qt designer:
# qt5-tools designer

# to convert the ui file into a py file:
# pyuic5 -x <path_to_ui_file.ui> -o <path_to_save.py>

import pandas as pd
from mplwidget import MplWidget
from PyQt5 import QtCore, QtGui, QtWidgets
from ma.utils import (
    load_participant_data,
    segment_participant_data_hr,
    segment_participant_data_rr,
    bandpass_filter,
)
import pathlib
import pickle
from copy import deepcopy
import os
from labeler_utils import (
    get_examples,
    change_visibilities_after_import_button,
    retranslateUi,
    init_visibilities,
    initUiElements,
    deactivite_radios,
    set_radios,
)

DATA_DIR = pathlib.Path("data/labeled_data/").resolve()


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setObjectName("MainWindow")
        self.resize(1150, 530)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")

        initUiElements(obj=self)

        for i in range(0, 24):
            if i < 2:
                self.comboBox_3.addItem("")
            if i < 21:
                self.comboBox.addItem("")
            self.comboBox_2.addItem("")

        init_visibilities(obj=self)

        self.importButton.clicked.connect(self.import_data)
        self.nextButton.clicked.connect(self.next_button_pushed)
        self.prevButton.clicked.connect(self.prev_button_pushed)
        self.prevButton_2.clicked.connect(self.prev_button_2_pushed)
        self.debugButton.clicked.connect(self.debug_pickle_labels)
        self.saveButton.clicked.connect(self.save_button_pushed)
        self.saveimageButton.clicked.connect(self.save_image_button_pushed)
        self.examplesButton.clicked.connect(self.examples_button_pushed)
        self.checkBox.stateChanged.connect(self.checkbox_state_changed)
        self.checkBox_2.stateChanged.connect(self.checkbox_2_state_changed)
        self.checkBox_3.stateChanged.connect(self.checkbox_3_state_changed)
        self.checkBox_4.stateChanged.connect(self.checkbox_4_state_changed)

        retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.centralwidget.setFocusPolicy(QtCore.Qt.StrongFocus)

    def keyPressEvent(self, event):
        match event.text():
            case "4":
                self.goodRadio.toggle()#
            case "p":
                self.goodRadio.toggle()
            case "5":
                self.badRadio.toggle()
            case "ü":
                self.badRadio.toggle()
            case "6":
                self.noisyRadio.toggle()
            case "+":
                self.noisyRadio.toggle()
            case "2":
                self.maRadio.toggle()
            case "ä":
                self.maRadio.toggle()
            case "1":
                self.noMaRadio.toggle()
            case "ö":
                self.noMaRadio.toggle()
            case "-":  # Unknown
                self.unknownMaRadio.toggle()
                self.unknownQualityRadio.toggle()
            case "+":  # Unknown
                self.unknownMaRadio.toggle()
                self.unknownQualityRadio.toggle()
            case "\r":  # Enter
                self.next_button_pushed()
            case "\x7f":  # Del
                self.prev_button_pushed()
            case "\x08":  # Backspace
                self.prev_button_2_pushed()
            case "f":
                if not self.checkBox.isChecked():
                    self.checkBox.setChecked(True)
                else:
                    self.checkBox.setChecked(False)
            case "a":
                if not self.checkBox_2.isChecked():
                    self.checkBox_2.setChecked(True)
                else:
                    self.checkBox_2.setChecked(False)
            case "t":
                if not self.checkBox_3.isChecked():
                    self.checkBox_3.setChecked(True)
                else:
                    self.checkBox_3.setChecked(False)
            case "b":
                if not self.checkBox_4.isChecked():
                    self.checkBox_4.setChecked(True)
                else:
                    self.checkBox_4.setChecked(False)

    def checkbox_state_changed(self):
        self.plot_signal()
        # self.plot_birds_eye_signal()

    def checkbox_2_state_changed(self):
        self.plot_signal()
        self.plot_birds_eye_signal()

    def checkbox_3_state_changed(self):
        self.plot_signal()
        self.plot_birds_eye_signal()

    def checkbox_4_state_changed(self):
        if self.checkBox_4.isChecked():
            self.MplWidget2.setVisible(True)
            self.resize(1200, 950)
        else:
            self.resize(1100, 530)
            self.MplWidget2.setVisible(False)
            self.plot_birds_eye_signal()

    def prev_button_2_pushed(self):
        # keep labels
        if self._signal_no == 0:
            return

        self._signal_no -= 1
        self.prevButton.setDisabled(True)
        self.saveButton.setDisabled(True)
        set_radios(self)
        deactivite_radios(self)

        self.plot_signal()
        if self.checkBox_4.isChecked():
            self.plot_birds_eye_signal()

    def examples_button_pushed(self):
        get_examples(obj=self)

    def save_image_button_pushed(self):
        self._participant_number = self.comboBox.currentText()
        signal_type = self.comboBox_2.currentText()
        self._class_type = self.comboBox_3.currentText()
        filename_base = f"Parti_{self._participant_number}_signaltype_{signal_type}_classtype_{self._class_type}_signalno_{self._signal_no}"
        # Save as PNG
        self.MplWidget.save_plot(filename_base, file_format="png")

        # Save as PGF for LaTeX
        self.MplWidget.save_plot(filename_base, file_format="pgf")

    def save_button_pushed(self):
        # End Session
        folder_name = "participant" + self._participant_number + "/tmp/"
        if not os.path.exists(DATA_DIR / folder_name):
            os.makedirs(DATA_DIR / folder_name)

        file_name = (
            self._signal_name.upper()
            + "_first_label_no_"
            + str(self._signal_start_no)
            + "_last_label_no_"
            + str(self._signal_no + 1)
            + "_"
            + self._class_type
            + ".xlsx"
        )
        full_path = DATA_DIR / folder_name / file_name

        # Save as a xlsx file:
        df = pd.DataFrame(self._labels)
        df.to_excel(full_path, index=False)

    def prev_button_pushed(self):
        # delete labels
        if self._signal_no == 0:
            return

        # Remove the last selected labels:
        for key, value in self._labels.items():
            self._labels[key] = value[:-1]

        self._signal_no -= 1
        self.plot_signal()
        if self.checkBox_4.isChecked():
            self.plot_birds_eye_signal()

    def next_button_pushed(self):
        if self.check_labels_selected() is False:
            return None
        self.save_selected_labels()
        self.reset_selected_radios()

        self._signal_no += 1

        # Save at each 100 label
        if self._signal_no % 100 == 0:
            self.save_button_pushed()
        # Finished:
        if self._signal_no >= self._number_of_signals:
            if self._signal_no == self._number_of_signals:
                # Export labels
                self.export_labels()
            self.label_2.setText(
                "You labeled the current signal.\n BEFORE CLOSING please check if the excel sheet saved.\n If the signal is not saved correctly, please click the debug button to pickle the labels. "
            )
            return None
        self.plot_signal()
        if self.checkBox_4.isChecked():
            self.plot_birds_eye_signal()

    def check_labels_selected(self) -> bool:
        if (
            self.qualityGroup.checkedButton() is None
            or self.qualityGroup.checkedButton().text() == ""
            or self.maGroup.checkedButton() is None
            or self.maGroup.checkedButton().text() == ""
        ):
            self.label_2.setText("YOU SHOULD SELECT THE LABELS ")
            self.label_2.adjustSize()
            self.label_2.setVisible(True)
            return False
        return True

    def reset_selected_radios(self):
        # If in previous mode, dont reset the radios:
        if len(self._labels["segment"]) > self._signal_no + 1:
            return
        self.invisibleQualityRadio.toggle()
        self.invisibleMaRadio.toggle()
        self.label_2.setVisible(False)

    def save_selected_labels(self):
        # Comes back from previous_button_2 with next button-> dont save anything:
        if len(self._labels["segment"]) > self._signal_no:
            set_radios(self, signal_no=self._signal_no + 1)
            return

        quality = self.qualityGroup.checkedButton().text()
        ma = self.maGroup.checkedButton().text()
        match quality:
            case "Good":
                quality_label = 0
            case "Bad":
                quality_label = 1
            case "Noisy":
                quality_label = 2
            case "?":
                quality_label = -1
        match ma:
            case "MA":
                ma_label = 1
            case "NOT MA":
                ma_label = 0
            case "?":
                ma_label = -1

        self._labels["segment"].append(self._signal_no + 1)
        self._labels["signalquality"].append(quality_label)
        self._labels["artifacts"].append(ma_label)

    def plot_signal(self):
        # Get the data:
        filter_checked = self.checkBox.isChecked()
        align_checked = self.checkBox_2.isChecked()
        twinx_checked = self.checkBox_3.isChecked()

        if filter_checked:
            data = self._filtered_data[self._signal_no]
        else:
            data = self._data[self._signal_no]

        # Get the reference data:
        data_ref = self._data_ref[self._signal_no].copy()
        if align_checked:
            diff = data.values[0] - data_ref.values[0]
            data_ref[data_ref.columns[0]] = data_ref.values + diff

        self._signal_name = data.columns[0]

        # Plot the data:
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.plot(data, label=self._signal_name)

        # Handle twin axis:
        if twinx_checked:
            if self._twin_ax is None:
                self._twin_ax = self.MplWidget.canvas.axes.twinx()
            self._twin_ax.clear()
            self._twin_ax.plot(
                data_ref, alpha=0.3, label=self._ref_signal_name.replace("_", " "), color="orange"
            )
            self._twin_ax.grid()
            self._twin_ax.legend(loc="upper right")
        else:
            if self._twin_ax is not None:
                self._twin_ax.clear()
                self._twin_ax.remove()
                self._twin_ax = None
            self.MplWidget.canvas.axes.plot(
                data_ref, alpha=0.3, label=self._ref_signal_name.replace("_", " ")
            )

        self.MplWidget.canvas.axes.set_title(
            f"Participant No: {self._participant_number} | Signal Type: {self._signal_name} | Signal No: {self._signal_no + 1} "
        )
        self.MplWidget.canvas.axes.set_xlabel("Time [s]")
        if self._signal_name.startswith("ecg"):
                y_label = "Voltage [mV]"
        elif self._signal_name.startswith("ppg"):
                y_label = "Voltage [mV]"
        elif self._signal_name.startswith("scg"):
                y_label = "Acceleration [m/s^2]"
        elif self._signal_name.startswith("mi"):
                y_label = "Tesla [T]"
        self.MplWidget.canvas.axes.set_ylabel(y_label)
        self.MplWidget.canvas.axes.grid()
        self.MplWidget.canvas.axes.legend(loc="upper left")
        self.MplWidget.canvas.draw()

    def plot_birds_eye_signal(self) -> None:
        try:
            birds_eye_amount = int(self.lineEdit_2.text())
        except:
            birds_eye_amount = 10
        # Get the data:
        # filter_checked = self.checkBox.isChecked()
        align_checked = self.checkBox_2.isChecked()
        twinx_checked = self.checkBox_3.isChecked()

        first_segment = max(0, self._signal_no - birds_eye_amount)
        last_segment = min(self._number_of_signals, self._signal_no + birds_eye_amount)

        # if filter_checked:
        #     data = self._filtered_data[self._signal_no]
        #     data_bird = self._filtered_data[first_segment:last_segment]
        # else:
        data = self._filtered_data[self._signal_no]
        data_bird = self._filtered_data[first_segment:last_segment]

        # Get the reference data:
        data_ref = self._data_ref[self._signal_no].copy()
        data_ref_bird = deepcopy(self._data_ref[first_segment:last_segment])
        if align_checked:
            diff = data.values[0] - data_ref.values[0]
            data_ref[data_ref.columns[0]] = data_ref.values + diff
            for data_ref_b, data_b in zip(data_ref_bird, data_bird):
                diff = data_b.values[0] - data_ref_b.values[0]
                data_ref_b[data_ref.columns[0]] = data_ref_b.values + diff

        self._signal_name = data.columns[0]

        # Plot the data:
        self.MplWidget2.canvas.axes.clear()
        [self.MplWidget2.canvas.axes.plot(data_b, color="blue") for data_b in data_bird]
        self.MplWidget2.canvas.axes.plot(data, label=self._signal_name, color="red")

        # Handle twin axis:
        if twinx_checked:
            if self._twin_ax_b is None:
                self._twin_ax_b = self.MplWidget2.canvas.axes.twinx()
            self._twin_ax_b.clear()
            [
                self._twin_ax_b.plot(
                    data_ref_b, alpha=0.3, label=self._ref_signal_name, color="orange"
                )
                for data_ref_b in data_ref_bird
            ]

            self._twin_ax_b.grid()
            self._twin_ax.legend(loc="upper right")
        else:
            if self._twin_ax_b is not None:
                self._twin_ax_b.clear()
                self._twin_ax_b.remove()
                self._twin_ax_b = None

            [
                self.MplWidget2.canvas.axes.plot(
                    data_ref_b, alpha=0.3, label=self._ref_signal_name, color="orange"
                )
                for data_ref_b in data_ref_bird
            ]

        self.MplWidget2.canvas.axes.set_xlabel("Time [s]")
        if self._signal_name.startswith("ecg"):
                y_label = "Voltage [mV]"
        elif self._signal_name.startswith("ppg"):
                y_label = "Voltage [mV]"
        elif self._signal_name.startswith("scg"):
                y_label = "Acceleration [m/s^2]"
        elif self._signal_name.startswith("mi"):
                y_label = "Tesla [T]"
        self.MplWidget2.canvas.axes.set_ylabel(y_label)
        self.MplWidget2.canvas.axes.set_title(
            f"Participant No: {self._participant_number} | Signal Type: {self._signal_name} | Signal No: {self._signal_no + 1} "
        )
        self.MplWidget2.canvas.axes.grid()
        self.MplWidget.canvas.axes.legend(loc="upper left")
        self.MplWidget2.canvas.draw()

    def import_data(self):
        # Import The Data:
        self._participant_number = self.comboBox.currentText()
        signal_type = self.comboBox_2.currentText()
        self._class_type = self.comboBox_3.currentText()
        df_participant = load_participant_data(
            participant_number=int(self._participant_number)
        )
        try:
            self._signal_start_no = int(self.lineEdit.text())
        except:
            self._signal_start_no = 1

        # Segment the signals based on the chosen class:
        df_signal = pd.DataFrame(df_participant[signal_type])
        df_signal.index = (df_signal.index - df_signal.index[0]).total_seconds()
        match self._class_type:
            case "HR":
                # Segmented data
                self._data = segment_participant_data_hr(df_participant=df_signal)
                # Reference signal
                self._ref_signal_name = "ecg_ref"
                df_signal_ref = pd.DataFrame(df_participant[self._ref_signal_name])
                df_signal_ref.index = (
                    df_signal_ref.index - df_signal_ref.index[0]
                ).total_seconds()
                self._data_ref = segment_participant_data_hr(
                    df_participant=df_signal_ref
                )
                # Filtered data
                df_filtered_data = bandpass_filter(
                    df_participant=df_signal, class_name="HR"
                )
                self._filtered_data = segment_participant_data_hr(
                    df_participant=df_filtered_data
                )

            case "RR":
                # Segmented data
                self._data = segment_participant_data_rr(df_participant=df_signal)
                # Reference signal
                self._ref_signal_name = "resp_ref"
                df_signal_ref = pd.DataFrame(df_participant[self._ref_signal_name])
                df_signal_ref.index = (
                    df_signal_ref.index - df_signal_ref.index[0]
                ).total_seconds()
                self._data_ref = segment_participant_data_rr(
                    df_participant=df_signal_ref
                )
                # Filtered data:
                df_filtered_data = bandpass_filter(
                    df_participant=df_signal, class_name="RR"
                )
                self._filtered_data = segment_participant_data_rr(
                    df_participant=df_filtered_data
                )

        self._number_of_signals = len(self._data)
        self._twin_ax = None
        self._twin_ax_b = None

        # Empty Labels:
        self._labels = {"segment": [], "signalquality": [], "artifacts": []}

        # UI Configurations:
        change_visibilities_after_import_button(obj=self)

        self.examples = True

        # Plot the first signal:
        self._signal_no = self._signal_start_no - 1
        self.plot_signal()

    def debug_pickle_labels(self):
        """
        Please use this function as a last debug chance in the session!
        """
        data = deepcopy(self._labels)
        folder_name = "participant" + self._participant_number
        if not os.path.exists(DATA_DIR / folder_name):
            os.makedirs(DATA_DIR / folder_name)

        file_name = self._signal_name.upper() + "_" + self._class_type + ".pickle"
        full_path = DATA_DIR / folder_name / file_name
        with open(full_path, "wb") as f:
            pickle.dump(data, f)

    def export_labels(self):
        # End Session
        folder_name = "participant" + self._participant_number
        if not os.path.exists(DATA_DIR / folder_name):
            os.makedirs(DATA_DIR / folder_name)

        file_name = self._signal_name.upper() + "_" + self._class_type + ".xlsx"
        full_path = DATA_DIR / folder_name / file_name

        # Save as a xlsx file:
        df = pd.DataFrame(self._labels)
        df.to_excel(full_path, index=False)

        # Closure:
        self.nextButton.setEnabled(False)
        self.label_2.setGeometry(QtCore.QRect(300, 310, 153, 31))
        self.label_2.setVisible(True)
        self.label_2.adjustSize()
        self.debugButton.setVisible(True)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = Ui_MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())
