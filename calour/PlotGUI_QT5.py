import matplotlib
matplotlib.use("Qt5Agg")
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QHBoxLayout, QVBoxLayout, QSizePolicy, QWidget, QPushButton, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from calour.heatmap import PlotGUI


app_ref=set()


class PlotGUI_QT5(PlotGUI):
    '''QT5 version of plot winfow GUI

    We open the figure as a widget inside the qt5 window
    '''
    def get_figure(self, newfig=None):
        self.aw = ApplicationWindow()
        app_ref.add(self.aw)
        self.aw.setWindowTitle("Calour")
        self.aw.show()
        return self.aw.plotfigure

    def update_info(self, taxname):
        self.aw.taxaLabel.setText(taxname)


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class ApplicationWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        self.main_widget = QWidget(self)

        l2 = QVBoxLayout()
        okButton = QPushButton("OK")
        cancelButton = QPushButton("Cancel")
        self.taxaLabel = QLabel()
        self.taxaLabel.setText('NA')
        self.taxaLabel.setFixedSize(200, 15)
        l2.addWidget(okButton)
        l2.addWidget(cancelButton)
        l2.addWidget(self.taxaLabel)

        l = QHBoxLayout(self.main_widget)
        sc = MyMplCanvas(self.main_widget, width=5, height=4, dpi=100)
        l.addWidget(sc)
        l.addLayout(l2)

        sc.setFocusPolicy(QtCore.Qt.ClickFocus)
        sc.setFocus()

        self.plotaxes = sc.axes
        self.plotfigure = sc.figure

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()
