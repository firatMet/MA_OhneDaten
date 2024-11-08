# ------------------------------------------------------
# -------------------- mplwidget.py --------------------
# ------------------------------------------------------
from PyQt5.QtWidgets import *

from matplotlib.backends.backend_qt5agg import FigureCanvas

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pickle


class MplWidget(QWidget):

    def __init__(self, parent=None):

        QWidget.__init__(self, parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)

        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)

    def save_plot(self, filename, file_format="png"):
        if file_format == "pgf":
            # Ensure that the necessary LaTeX packages are used
            try:
                plt.rcParams["pgf.preamble"] = r"\usepackage{tikz}"
            except:
                raise KeyError(
                    "Please be sure that a TeX system is installed on your machine. Preferably MiKTeX for Windows."
                )
        self.figure.savefig(f"{filename}.{file_format}", format=file_format, dpi=300)
        with open(f'{filename}.pkl', 'wb') as f:
            pickle.dump(self.figure, f)
