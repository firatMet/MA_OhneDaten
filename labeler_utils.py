import pathlib
import re
from PyQt5 import QtGui, QtCore, QtWidgets
from mplwidget import MplWidget

EXAMPLES_DIR = pathlib.Path("data/examples/").resolve()


def set_radios(obj, signal_no: int = None) -> None:
    if not signal_no:
        signal_no = obj._signal_no
    try:
        signalquality = obj._labels["signalquality"][signal_no]
        ma = obj._labels["artifacts"][signal_no]
    except:  # If we come back where we left:
        activate_radios(obj=obj)
        obj.prevButton.setDisabled(False)
        obj.saveButton.setDisabled(False)
        return
    match signalquality:
        case 0:
            obj.goodRadio.toggle()
        case 1:
            obj.badRadio.toggle()
        case 2:
            obj.noisyRadio.toggle()
        case -1:
            obj.unknownQualityRadio.toggle()

    match ma:
        case 0:
            obj.noMaRadio.toggle()
        case 1:
            obj.maRadio.toggle()
        case -1:
            obj.unknownMaRadio.toggle()


def activate_radios(obj) -> None:
    obj.goodRadio.setDisabled(False)
    obj.badRadio.setDisabled(False)
    obj.noisyRadio.setDisabled(False)
    obj.unknownQualityRadio.setDisabled(False)
    obj.maRadio.setDisabled(False)
    obj.noMaRadio.setDisabled(False)
    obj.unknownMaRadio.setDisabled(False)


def deactivite_radios(obj) -> None:
    obj.goodRadio.setDisabled(True)
    obj.badRadio.setDisabled(True)
    obj.noisyRadio.setDisabled(True)
    obj.unknownQualityRadio.setDisabled(True)
    obj.maRadio.setDisabled(True)
    obj.noMaRadio.setDisabled(True)
    obj.unknownMaRadio.setDisabled(True)


def initUiElements(obj) -> None:
    # Filter
    obj.checkBox = QtWidgets.QCheckBox(obj.centralwidget)
    obj.checkBox.setGeometry(QtCore.QRect(290, -10, 80, 50))
    obj.checkBox.setObjectName("checkBox")

    # Align signal & reference signal
    obj.checkBox_2 = QtWidgets.QCheckBox(obj.centralwidget)
    obj.checkBox_2.setGeometry(QtCore.QRect(380, -10, 80, 50))
    obj.checkBox_2.setObjectName("checkBox_2")

    # Twin Axes
    obj.checkBox_3 = QtWidgets.QCheckBox(obj.centralwidget)
    obj.checkBox_3.setGeometry(QtCore.QRect(480, -10, 90, 50))
    obj.checkBox_3.setObjectName("checkBox_3")

    # Bird's Eye View
    obj.checkBox_4 = QtWidgets.QCheckBox(obj.centralwidget)
    obj.checkBox_4.setGeometry(QtCore.QRect(580, -10, 130, 50))
    obj.checkBox_4.setObjectName("checkBox_4")

    obj.lineEdit_2 = QtWidgets.QLineEdit(obj.centralwidget)
    obj.lineEdit_2.setGeometry(QtCore.QRect(715, 7, 50, 16))
    obj.lineEdit_2.setText("50")
    obj.lineEdit_2.setObjectName("lineEdit")

    # To reset the radio button groups, when the next button is clicked.
    obj.invisibleQualityRadio = QtWidgets.QRadioButton(obj.centralwidget)
    obj.invisibleQualityRadio.setGeometry(QtCore.QRect(920, 120, 95, 20))
    obj.invisibleQualityRadio.setObjectName("invisibleQualityRadio")
    obj.invisibleMaRadio = QtWidgets.QRadioButton(obj.centralwidget)
    obj.invisibleMaRadio.setGeometry(QtCore.QRect(920, 120, 95, 20))
    obj.invisibleMaRadio.setObjectName("invisibleMaRadio")

    obj.nextButton = QtWidgets.QPushButton(obj.centralwidget)
    obj.nextButton.setGeometry(QtCore.QRect(920, 160, 153, 31))
    obj.nextButton.setObjectName("nextButton")
    obj.debugButton = QtWidgets.QPushButton(obj.centralwidget)
    obj.debugButton.setGeometry(QtCore.QRect(950, 310, 153, 31))
    obj.debugButton.setObjectName("debugButton")
    obj.prevButton = QtWidgets.QPushButton(obj.centralwidget)
    obj.prevButton.setGeometry(QtCore.QRect(290, 370, 200, 31))
    obj.setObjectName("prevButton")
    obj.prevButton_2 = QtWidgets.QPushButton(obj.centralwidget)
    obj.prevButton_2.setGeometry(QtCore.QRect(540, 370, 200, 31))
    obj.prevButton_2.setObjectName("prevButton_2")
    obj.saveButton = QtWidgets.QPushButton(obj.centralwidget)
    obj.saveButton.setGeometry(QtCore.QRect(920, 370, 93, 31))
    obj.saveButton.setObjectName("saveButton")
    obj.saveimageButton = QtWidgets.QPushButton(obj.centralwidget)
    obj.saveimageButton.setGeometry(QtCore.QRect(1050, 370, 93, 31))
    obj.saveimageButton.setObjectName("saveimageButton")
    obj.importButton = QtWidgets.QPushButton(obj.centralwidget)
    obj.importButton.setGeometry(QtCore.QRect(920, 310, 93, 28))
    obj.importButton.setObjectName("importButton")
    obj.examplesButton = QtWidgets.QPushButton(obj.centralwidget)
    obj.examplesButton.setGeometry(QtCore.QRect(920, 310, 93, 31))
    obj.examplesButton.setObjectName("examplesButton")

    obj.goodRadio = QtWidgets.QRadioButton(obj.centralwidget)
    obj.goodRadio.setGeometry(QtCore.QRect(920, 80, 95, 20))
    obj.goodRadio.setObjectName("goodRadio")
    obj.badRadio = QtWidgets.QRadioButton(obj.centralwidget)
    obj.badRadio.setGeometry(QtCore.QRect(970, 80, 95, 20))
    obj.badRadio.setObjectName("badRadio")
    obj.noisyRadio = QtWidgets.QRadioButton(obj.centralwidget)
    obj.noisyRadio.setGeometry(QtCore.QRect(1020, 80, 95, 20))
    obj.noisyRadio.setObjectName("noisyRadio")
    obj.unknownQualityRadio = QtWidgets.QRadioButton(obj.centralwidget)
    obj.unknownQualityRadio.setGeometry(QtCore.QRect(1070, 80, 95, 20))
    obj.unknownQualityRadio.setObjectName("unknownQualityRadio")

    obj.maRadio = QtWidgets.QRadioButton(obj.centralwidget)
    obj.maRadio.setGeometry(QtCore.QRect(1000, 100, 95, 20))
    obj.maRadio.setObjectName("MA")
    obj.noMaRadio = QtWidgets.QRadioButton(obj.centralwidget)
    obj.noMaRadio.setGeometry(QtCore.QRect(920, 100, 75, 20))
    obj.noMaRadio.setObjectName("NOT MA")
    obj.unknownMaRadio = QtWidgets.QRadioButton(obj.centralwidget)
    obj.unknownMaRadio.setGeometry(QtCore.QRect(1070, 100, 95, 20))
    obj.unknownMaRadio.setObjectName("unknownQualityRadio")

    obj.qualityGroup = QtWidgets.QButtonGroup(obj)
    obj.qualityGroup.setObjectName("quailtyGroup")
    obj.qualityGroup.addButton(obj.goodRadio)
    obj.qualityGroup.addButton(obj.badRadio)
    obj.qualityGroup.addButton(obj.noisyRadio)
    obj.qualityGroup.addButton(obj.unknownQualityRadio)
    obj.qualityGroup.addButton(obj.invisibleQualityRadio)

    obj.maGroup = QtWidgets.QButtonGroup(obj)
    obj.maGroup.setObjectName("maGroup")
    obj.maGroup.addButton(obj.maRadio)
    obj.maGroup.addButton(obj.noMaRadio)
    obj.maGroup.addButton(obj.unknownMaRadio)
    obj.maGroup.addButton(obj.invisibleMaRadio)

    obj.lineEdit = QtWidgets.QLineEdit(obj.centralwidget)
    obj.lineEdit.setGeometry(QtCore.QRect(920, 290, 50, 16))
    obj.lineEdit.setText("1")
    obj.lineEdit.setObjectName("lineEdit")

    obj.label = QtWidgets.QLabel(obj.centralwidget)
    obj.label.setGeometry(QtCore.QRect(920, 170, 151, 16))
    obj.label.setObjectName("label")
    obj.label_2 = QtWidgets.QLabel(obj.centralwidget)
    obj.label_2.setGeometry(QtCore.QRect(920, 220, 161, 16))
    obj.label_2.setObjectName("label_2")
    obj.label_3 = QtWidgets.QLabel(obj.centralwidget)
    obj.label_3.setGeometry(QtCore.QRect(920, 270, 251, 16))
    obj.label_3.setObjectName("label_3")
    obj.label_17 = QtWidgets.QLabel(obj.centralwidget)
    obj.label_17.setGeometry(QtCore.QRect(920, 120, 251, 16))
    obj.label_17.setObjectName("label_17")

    # Example Pictures:
    obj.label_4 = QtWidgets.QLabel(obj.centralwidget)
    obj.label_4.setObjectName("label_4")
    obj.label_4.setGeometry(QtCore.QRect(250, 400, 95, 20))
    obj.label_5 = QtWidgets.QLabel(obj.centralwidget)
    obj.label_5.setObjectName("label_5")
    obj.label_5.setGeometry(QtCore.QRect(775, 400, 95, 20))
    obj.label_6 = QtWidgets.QLabel(obj.centralwidget)
    obj.label_6.setObjectName("label_6")
    obj.label_6.setGeometry(QtCore.QRect(1325, 400, 95, 20))
    obj.label_13 = QtWidgets.QLabel(obj.centralwidget)
    obj.label_13.setObjectName("label_13")
    obj.label_13.setGeometry(QtCore.QRect(250, 700, 95, 20))
    obj.label_14 = QtWidgets.QLabel(obj.centralwidget)
    obj.label_14.setObjectName("label_14")
    obj.label_14.setGeometry(QtCore.QRect(775, 700, 95, 20))
    obj.label_15 = QtWidgets.QLabel(obj.centralwidget)
    obj.label_15.setObjectName("label_15")
    obj.label_15.setGeometry(QtCore.QRect(1325, 700, 95, 20))

    obj.label_7 = QtWidgets.QLabel(obj.centralwidget)
    obj.label_7.setObjectName("label_7")
    obj.label_7.setGeometry(QtCore.QRect(550, 375, 95, 20))
    obj.label_8 = QtWidgets.QLabel(obj.centralwidget)
    obj.label_8.setObjectName("label_8")
    obj.label_8.setGeometry(QtCore.QRect(1100, 375, 95, 20))
    obj.label_9 = QtWidgets.QLabel(obj.centralwidget)
    obj.label_9.setObjectName("label_9")
    obj.label_9.setGeometry(QtCore.QRect(1650, 375, 95, 20))
    obj.label_10 = QtWidgets.QLabel(obj.centralwidget)
    obj.label_10.setObjectName("label_10")
    obj.label_10.setGeometry(QtCore.QRect(550, 675, 95, 20))
    obj.label_11 = QtWidgets.QLabel(obj.centralwidget)
    obj.label_11.setObjectName("label_11")
    obj.label_11.setGeometry(QtCore.QRect(1100, 675, 95, 20))
    obj.label_12 = QtWidgets.QLabel(obj.centralwidget)
    obj.label_12.setObjectName("label_12")
    obj.label_12.setGeometry(QtCore.QRect(1650, 675, 95, 20))

    obj.label_16 = QtWidgets.QLabel(obj.centralwidget)
    obj.label_16.setObjectName("label_16")
    obj.label_16.setGeometry(QtCore.QRect(1100, 100, 95, 20))

    obj.comboBox = QtWidgets.QComboBox(obj.centralwidget)
    obj.comboBox.setGeometry(QtCore.QRect(920, 190, 73, 22))
    obj.comboBox.setObjectName("comboBox")

    obj.comboBox_2 = QtWidgets.QComboBox(obj.centralwidget)
    obj.comboBox_2.setGeometry(QtCore.QRect(920, 240, 73, 22))
    obj.comboBox_2.setObjectName("comboBox_2")

    obj.comboBox_3 = QtWidgets.QComboBox(obj.centralwidget)
    obj.comboBox_3.setGeometry(QtCore.QRect(920, 140, 73, 22))
    obj.comboBox_3.setObjectName("comboBox_3")

    obj.MplWidget = MplWidget(obj.centralwidget)
    obj.MplWidget.setGeometry(QtCore.QRect(100, 20, 800, 500))
    obj.MplWidget.setObjectName("MplWidget")

    # Bird's Eye
    obj.MplWidget2 = MplWidget(obj.centralwidget)
    obj.MplWidget2.setGeometry(QtCore.QRect(70, 500, 1100, 400))
    obj.MplWidget2.setObjectName("MplWidget")
    (670, 310, 93, 31)

    obj.setCentralWidget(obj.centralwidget)
    obj.menubar = QtWidgets.QMenuBar(obj)
    obj.menubar.setGeometry(QtCore.QRect(0, 0, 835, 26))
    obj.menubar.setObjectName("menubar")
    obj.setMenuBar(obj.menubar)
    obj.statusbar = QtWidgets.QStatusBar(obj)
    obj.statusbar.setObjectName("statusbar")
    obj.setStatusBar(obj.statusbar)


def retranslateUi(obj) -> None:
    _translate = QtCore.QCoreApplication.translate
    obj.setWindowTitle(_translate("MainWindow", "MainWindow"))
    obj.nextButton.setText(_translate("MainWindow", "Next"))
    obj.prevButton.setText(_translate("MainWindow", "Previous (delete labels) (DEL)"))
    obj.prevButton_2.setText(_translate("MainWindow", "Previous (keep labels) (<--)"))
    obj.debugButton.setText(_translate("MainWindow", "DEBUG-PICKLE"))
    obj.saveButton.setText(_translate("MainWindow", "Instant-Save"))
    obj.saveimageButton.setText(_translate("MainWindow", "Save Image"))
    obj.examplesButton.setText(_translate("MainWindow", "Show Examples"))
    obj.goodRadio.setText(_translate("MainWindow", "Good"))
    obj.badRadio.setText(_translate("MainWindow", "Bad"))
    obj.unknownQualityRadio.setText(_translate("MainWindow", "?"))
    obj.noisyRadio.setText(_translate("MainWindow", "Noisy"))  #
    obj.maRadio.setText(_translate("MainWindow", "MA"))
    obj.noMaRadio.setText(_translate("MainWindow", "NOT MA"))
    obj.unknownMaRadio.setText(_translate("MainWindow", "?"))
    obj.checkBox.setText(_translate("MainWindow", "Filter (F)"))
    obj.checkBox_2.setText(_translate("MainWindow", "Align (A)"))
    obj.checkBox_3.setText(_translate("MainWindow", "TwinAx (T)"))
    obj.checkBox_4.setText(_translate("MainWindow", "Bird's Eye View (B)"))
    obj.label.setText(_translate("MainWindow", "Please select the participant number:"))
    obj.label.adjustSize()
    obj.comboBox.setItemText(0, _translate("MainWindow", "1"))
    obj.comboBox.setItemText(1, _translate("MainWindow", "2"))
    obj.comboBox.setItemText(2, _translate("MainWindow", "3"))
    obj.comboBox.setItemText(3, _translate("MainWindow", "4"))
    obj.comboBox.setItemText(4, _translate("MainWindow", "5"))
    obj.comboBox.setItemText(5, _translate("MainWindow", "6"))
    obj.comboBox.setItemText(6, _translate("MainWindow", "7"))
    obj.comboBox.setItemText(7, _translate("MainWindow", "8"))
    obj.comboBox.setItemText(8, _translate("MainWindow", "9"))
    obj.comboBox.setItemText(9, _translate("MainWindow", "10"))
    obj.comboBox.setItemText(10, _translate("MainWindow", "11"))
    obj.comboBox.setItemText(11, _translate("MainWindow", "12"))
    obj.comboBox.setItemText(12, _translate("MainWindow", "13"))
    obj.comboBox.setItemText(13, _translate("MainWindow", "14"))
    obj.comboBox.setItemText(14, _translate("MainWindow", "15"))
    obj.comboBox.setItemText(15, _translate("MainWindow", "16"))
    obj.comboBox.setItemText(16, _translate("MainWindow", "17"))
    obj.comboBox.setItemText(17, _translate("MainWindow", "18"))
    obj.comboBox.setItemText(18, _translate("MainWindow", "19"))
    obj.comboBox.setItemText(19, _translate("MainWindow", "20"))
    obj.importButton.setText(_translate("MainWindow", "Import Data"))
    obj.label_2.setText(
        _translate("MainWindow", "Please select the signal to segment:")
    )
    obj.label_2.adjustSize()
    obj.comboBox_2.setItemText(0, _translate("MainWindow", "ecg1"))
    obj.comboBox_2.setItemText(1, _translate("MainWindow", "ecg2"))
    obj.comboBox_2.setItemText(2, _translate("MainWindow", "ecg3"))
    obj.comboBox_2.setItemText(3, _translate("MainWindow", "ecg4"))
    obj.comboBox_2.setItemText(4, _translate("MainWindow", "ppg1"))
    obj.comboBox_2.setItemText(5, _translate("MainWindow", "ppg2"))
    obj.comboBox_2.setItemText(6, _translate("MainWindow", "ppg3"))
    obj.comboBox_2.setItemText(7, _translate("MainWindow", "ppg4"))
    obj.comboBox_2.setItemText(8, _translate("MainWindow", "scg1x"))
    obj.comboBox_2.setItemText(9, _translate("MainWindow", "scg1y"))
    obj.comboBox_2.setItemText(10, _translate("MainWindow", "scg1z"))
    obj.comboBox_2.setItemText(11, _translate("MainWindow", "scg2x"))
    obj.comboBox_2.setItemText(12, _translate("MainWindow", "scg2y"))
    obj.comboBox_2.setItemText(13, _translate("MainWindow", "scg2z"))
    obj.comboBox_2.setItemText(14, _translate("MainWindow", "scg3x"))
    obj.comboBox_2.setItemText(15, _translate("MainWindow", "scg3y"))
    obj.comboBox_2.setItemText(16, _translate("MainWindow", "scg3z"))
    obj.comboBox_2.setItemText(17, _translate("MainWindow", "scg4x"))
    obj.comboBox_2.setItemText(18, _translate("MainWindow", "scg4y"))
    obj.comboBox_2.setItemText(19, _translate("MainWindow", "scg4z"))
    obj.comboBox_2.setItemText(20, _translate("MainWindow", "mi1"))
    obj.comboBox_2.setItemText(21, _translate("MainWindow", "mi2"))
    obj.comboBox_2.setItemText(22, _translate("MainWindow", "mi3"))
    obj.comboBox_2.setItemText(23, _translate("MainWindow", "mi4"))
    obj.label_3.setText(
        _translate("MainWindow", "Please select the signal number to start:")
    )
    obj.label_3.adjustSize()

    obj.comboBox_3.setItemText(0, _translate("MainWindow", "RR"))
    obj.comboBox_3.setItemText(1, _translate("MainWindow", "HR"))
    obj.label_17.setText(_translate("MainWindow", "Please select the class type:"))
    obj.label_17.adjustSize()

    obj.label_7.setText(_translate("MainWindow", "Good HR:"))
    obj.label_8.setText(_translate("MainWindow", "Bad HR:"))
    obj.label_9.setText(_translate("MainWindow", "Noisy HR:"))

    obj.label_10.setText(_translate("MainWindow", "Good RR:"))
    obj.label_11.setText(_translate("MainWindow", "Bad RR:"))
    obj.label_12.setText(_translate("MainWindow", "Noisy RR:"))

    obj.label_16.setText(_translate("MainWindow", "Instructions"))


def init_visibilities(obj) -> None:
    obj.invisibleQualityRadio.setVisible(False)
    obj.invisibleMaRadio.setVisible(False)
    obj.nextButton.setVisible(False)
    obj.prevButton.setVisible(False)
    obj.prevButton_2.setVisible(False)
    obj.debugButton.setVisible(False)
    obj.saveButton.setVisible(False)
    obj.saveimageButton.setVisible(False)
    obj.examplesButton.setVisible(False)
    obj.goodRadio.setVisible(False)
    obj.badRadio.setVisible(False)
    obj.noisyRadio.setVisible(False)
    obj.unknownQualityRadio.setVisible(False)
    obj.maRadio.setVisible(False)
    obj.noMaRadio.setVisible(False)
    obj.unknownMaRadio.setVisible(False)
    obj.label_4.setVisible(False)
    obj.label_5.setVisible(False)
    obj.label_6.setVisible(False)
    obj.label_7.setVisible(False)
    obj.label_8.setVisible(False)
    obj.label_9.setVisible(False)
    obj.label_10.setVisible(False)
    obj.label_11.setVisible(False)
    obj.label_12.setVisible(False)
    obj.label_13.setVisible(False)
    obj.label_14.setVisible(False)
    obj.label_15.setVisible(False)
    obj.label_16.setVisible(False)
    obj.checkBox.setVisible(False)
    obj.checkBox_2.setVisible(False)
    obj.checkBox_3.setVisible(False)
    obj.checkBox_4.setVisible(False)
    obj.MplWidget2.setVisible(False)
    obj.lineEdit_2.setVisible(False)


def change_visibilities_after_import_button(obj) -> None:
    obj.nextButton.setVisible(True)
    obj.prevButton.setVisible(True)
    obj.prevButton_2.setVisible(True)
    obj.saveButton.setVisible(True)
    obj.saveimageButton.setVisible(True)
    obj.examplesButton.setVisible(False)
    obj.goodRadio.setVisible(True)
    obj.badRadio.setVisible(True)
    obj.noisyRadio.setVisible(True)
    obj.unknownQualityRadio.setVisible(True)
    obj.maRadio.setVisible(True)
    obj.noMaRadio.setVisible(True)
    obj.unknownMaRadio.setVisible(True)
    obj.checkBox.setVisible(True)
    obj.checkBox_2.setVisible(True)
    obj.checkBox_3.setVisible(True)
    obj.checkBox_4.setVisible(True)
    obj.importButton.setVisible(False)
    obj.label.setVisible(False)
    obj.label_2.setVisible(False)
    obj.label_3.setVisible(False)
    obj.label_17.setVisible(False)
    obj.comboBox.setVisible(False)
    obj.comboBox_2.setVisible(False)
    obj.comboBox_3.setVisible(False)
    obj.lineEdit.setVisible(False)
    obj.lineEdit_2.setVisible(True)


def get_examples(obj) -> None:
    signal_type = re.sub(r"\d+", "", obj.comboBox_2.currentText())

    match signal_type:
        case "ecg":
            pixmap_good_hr = QtGui.QPixmap(str(EXAMPLES_DIR / "good_ecg_hr.png"))
            pixmap_bad_hr = QtGui.QPixmap(str(EXAMPLES_DIR / "bad_ecg_hr.png"))
            pixmap_noisy_hr = QtGui.QPixmap(str(EXAMPLES_DIR / "noisy_ecg_hr.png"))
            pixmap_good_rr = QtGui.QPixmap(str(EXAMPLES_DIR / "good_ecg_rr.png"))
            pixmap_bad_rr = QtGui.QPixmap(str(EXAMPLES_DIR / "bad_ecg_rr.png"))
            pixmap_noisy_rr = QtGui.QPixmap(str(EXAMPLES_DIR / "noisy_ecg_rr.png"))
            instruction = """   HR
                        Good quality: clear peaks (R-peaks) visible. HR can be determined with a vanishing error
                        Bad quality: peaks visible, but HR can only be determined with a error, due to noise or artefacts
                        Noisy: No information about the HR can be determined
                        If an MA is present, the quality is good, if the HR information is completely contained, i.e. no peaks are masked by the MA an all peaks have a prominance with respect to noise
                                        
    RR
                        Good quality: A respiratory related amplitude modulation is clearly visible. The RR can be determined with vanishing error.
                        Bad quality: A respiratory related amplitude modulation is visible, but is distorted by noisy/artefacts. The RR can only be determined with an error
                        Noisy: No information about the RR is visible."""
        # TODO : Add the images here
        case "ppg":
            pixmap_good_hr = QtGui.QPixmap(str(EXAMPLES_DIR / "good_ppg_hr.png"))
            pixmap_bad_hr = QtGui.QPixmap(str(EXAMPLES_DIR / "bad_ppg_hr.png"))
            pixmap_noisy_hr = QtGui.QPixmap(str(EXAMPLES_DIR / "noisy_ppg_hr.png"))
            pixmap_good_rr = QtGui.QPixmap(str(EXAMPLES_DIR / "good_ppg_rr.png"))
            pixmap_bad_rr = QtGui.QPixmap(str(EXAMPLES_DIR / "bad_ppg_rr.png"))
            pixmap_noisy_rr = QtGui.QPixmap(str(EXAMPLES_DIR / "noisy_ppg_rr.png"))
            instruction = """   HR
                        Good quality: clear peaks visible. HR can be determined with a vanishing error
                        Bad quality: peaks visible, but HR can only be determined with a error, due to noise or artefacts
                        Noisy: No information about the HR can be determined
                                        
    RR
                        Good quality: A respiratory related amplitude modulation is clearly visible. The RR can be determined with vanishing error.
                        Bad quality: A respiratory related amplitude modulation is visible, but is distorted by noisy/artefacts. The RR can only be determined with an error
                        Noisy: No information about the RR is visible."""

        case "mi":
            pixmap_good_hr = QtGui.QPixmap(str(EXAMPLES_DIR / "good_mi_hr.png"))
            pixmap_bad_hr = QtGui.QPixmap(str(EXAMPLES_DIR / "bad_mi_hr.png"))
            pixmap_noisy_hr = QtGui.QPixmap(str(EXAMPLES_DIR / "noisy_mi_hr.png"))
            pixmap_good_rr = QtGui.QPixmap(str(EXAMPLES_DIR / "good_mi_rr.png"))
            pixmap_bad_rr = QtGui.QPixmap(str(EXAMPLES_DIR / "bad_mi_rr.png"))
            pixmap_noisy_rr = QtGui.QPixmap(str(EXAMPLES_DIR / "noisy_mi_rr.png"))
            instruction = """   HR
                                        -
                                        
    RR
                        Good quality: A respiratory related amplitude modulation is clearly visible. The RR can be determined with vanishing error.
                        Bad quality: A respiratory related amplitude modulation is visible, but is distorted by noisy/artefacts. The RR can only be determined with an error
                        Noisy: No information about the RR is visible.."""

        case "acc":
            pixmap_good_hr = QtGui.QPixmap(str(EXAMPLES_DIR / "good_acc_hr.png"))
            pixmap_bad_hr = QtGui.QPixmap(str(EXAMPLES_DIR / "bad_acc_hr.png"))
            pixmap_noisy_hr = QtGui.QPixmap(str(EXAMPLES_DIR / "noisy_acc_hr.png"))
            pixmap_good_rr = QtGui.QPixmap(str(EXAMPLES_DIR / "good_acc_rr.png"))
            pixmap_bad_rr = QtGui.QPixmap(str(EXAMPLES_DIR / "bad_acc_rr.png"))
            pixmap_noisy_rr = QtGui.QPixmap(str(EXAMPLES_DIR / "noisy_acc_rr.png"))
            instruction = """   HR
                        Good quality: HR related pattern clearly visible (with reference). HR can be determined with a vanishing error
                        Bad quality: HR related pattern not always clear, but HR can only be determined with a error.
                        Noisy: No information about the HR can be determined
                                        
    RR
                        Good quality: A respiratory related amplitude modulation is clearly visible. The RR can be determined with vanishing error.
                        Bad quality: A respiratory related amplitude modulation is visible, but is distorted by noisy/artefacts. The RR can only be determined with an error
                        Noisy: No information about the RR is visible."""
    obj.label_4.setPixmap(pixmap_good_hr)
    obj.label_5.setPixmap(pixmap_bad_hr)
    obj.label_6.setPixmap(pixmap_noisy_hr)
    obj.label_13.setPixmap(pixmap_good_rr)
    obj.label_14.setPixmap(pixmap_bad_rr)
    obj.label_15.setPixmap(pixmap_noisy_rr)
    obj.label_16.setText(instruction)

    obj.label_4.resize(pixmap_good_hr.width(), pixmap_good_hr.height())
    obj.label_5.resize(pixmap_bad_hr.width(), pixmap_bad_hr.height())
    obj.label_6.resize(pixmap_noisy_hr.width(), pixmap_noisy_hr.height())
    obj.label_13.resize(pixmap_good_rr.width(), pixmap_good_rr.height())
    obj.label_14.resize(pixmap_bad_rr.width(), pixmap_bad_rr.height())
    obj.label_15.resize(pixmap_noisy_rr.width(), pixmap_noisy_rr.height())
    obj.label_16.adjustSize()

    if obj.examples:
        obj.label_4.setVisible(True)
        obj.label_5.setVisible(True)
        obj.label_6.setVisible(True)
        obj.label_7.setVisible(True)
        obj.label_8.setVisible(True)
        obj.label_9.setVisible(True)
        obj.label_10.setVisible(True)
        obj.label_11.setVisible(True)
        obj.label_12.setVisible(True)
        obj.label_13.setVisible(True)
        obj.label_14.setVisible(True)
        obj.label_15.setVisible(True)
        obj.label_16.setVisible(True)
        obj.examples = False
        obj.resize(1700, 1000)
    else:
        obj.label_4.setVisible(False)
        obj.label_5.setVisible(False)
        obj.label_6.setVisible(False)
        obj.label_7.setVisible(False)
        obj.label_8.setVisible(False)
        obj.label_9.setVisible(False)
        obj.label_10.setVisible(False)
        obj.label_11.setVisible(False)
        obj.label_12.setVisible(False)
        obj.label_13.setVisible(False)
        obj.label_14.setVisible(False)
        obj.label_15.setVisible(False)
        obj.label_16.setVisible(False)
        obj.examples = True
        obj.resize(900, 400)
