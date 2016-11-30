import matplotlib
matplotlib.use("Qt5Agg")
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QHBoxLayout, QVBoxLayout, QSizePolicy, QWidget, QPushButton, QLabel, QListWidget, QSplitter, QFrame, QComboBox, QScrollArea, QListWidgetItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from calour.bactdb import BactDB
from calour.gui.plotgui import PlotGUI


# app_ref=set()


class PlotGUI_QT5(PlotGUI):
    '''QT5 version of plot winfow GUI

    We open the figure as a widget inside the qt5 window
    '''
    def __init__(self, *kargs, **kwargs):
        PlotGUI.__init__(self, *kargs, **kwargs)
        self.bactdb = BactDB()

    def get_figure(self, newfig=None):
        self.aw = ApplicationWindow(self.exp)
#        app_ref.add(self.aw)
        self.aw.setWindowTitle("Calour")
        self.aw.show()
        return self.aw.plotfigure

    def update_info(self):
        taxname = self.exp.feature_metadata['taxonomy'][self.last_select_feature]
        sequence = self.exp.feature_metadata.index[self.last_select_feature]
        self.aw.w_taxonomy.setText(taxname)
        self.aw.w_reads.setText('reads:{:.01f}'.format(self.exp.get_data()[self.last_select_sample, self.last_select_feature]))
        # self.aw.w_dblist.addItem(taxname)
        csample_field = str(self.aw.w_field.currentText())
        self.aw.w_field_val.setText(str(self.exp.sample_metadata[csample_field][self.last_select_sample]))

        self.aw.w_dblist.clear()
        info = self.bactdb.getannotationstrings(sequence)
        self.addtocdblist(info)

    def addtocdblist(self, info):
        """
        add to cdb list without clearing
        """
        for cinfo in info:
            details = cinfo[0]
            newitem = QListWidgetItem(cinfo[1])
            newitem.setData(QtCore.Qt.UserRole, details)
            if details['annotationtype'] == 'diffexp':
                ccolor = QtGui.QColor(0, 0, 200)
            elif details['annotationtype'] == 'contamination':
                ccolor = QtGui.QColor(200, 0, 0)
            elif details['annotationtype'] == 'common':
                ccolor = QtGui.QColor(0, 200, 0)
            elif details['annotationtype'] == 'highfreq':
                ccolor = QtGui.QColor(0, 200, 0)
            else:
                ccolor = QtGui.QColor(0, 0, 0)
            newitem.setForeground(ccolor)
            self.aw.w_dblist.addItem(newitem)


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
    def __init__(self, exp):
        QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        self.main_widget = QWidget(self)

        # set the GUI widgets
        # the left side (right side is the heatmap)
        userside = QVBoxLayout()
        # field to display
        lbox_field = QHBoxLayout()
        self.w_field = QComboBox()
        self.w_field_val = QLabel()
        self.w_field_val.setText('NA')
        lbox_field.addWidget(self.w_field)
        lbox_field.addWidget(self.w_field_val)
        userside.addLayout(lbox_field)
        # taxonomy
        lbox_tax = QHBoxLayout()
        taxlabel = QLabel(text='tax:')
        taxscroll = QScrollArea()
        taxscroll.setFixedHeight(18)
        self.w_taxonomy = QLabel(text='NA')
        taxscroll.setWidget(self.w_taxonomy)
        self.w_taxonomy.setMinimumWidth(800)
        lbox_tax.addWidget(taxlabel)
        lbox_tax.addWidget(taxscroll)
        userside.addLayout(lbox_tax)
        # reads
        lbox_reads = QHBoxLayout()
        readslabel = QLabel(text='reads:')
        self.w_reads = QLabel(text='?')
        lbox_reads.addWidget(readslabel)
        lbox_reads.addWidget(self.w_reads)
        userside.addLayout(lbox_reads)
        # buttons
        lbox_buttons = QHBoxLayout()
        self.w_sequence = QPushButton(text='Sequence')
        lbox_buttons.addWidget(self.w_sequence)
        self.w_info = QPushButton(text='Info')
        lbox_buttons.addWidget(self.w_info)
        self.w_annotate = QPushButton(text='Annotate')
        lbox_buttons.addWidget(self.w_annotate)
        userside.addLayout(lbox_buttons)
        # db annotations list
        self.w_dblist = QListWidget()
        userside.addWidget(self.w_dblist)

        layout = QHBoxLayout(self.main_widget)
        heatmap = MyMplCanvas(self.main_widget, width=5, height=4, dpi=100)
        frame = QFrame()
        splitter = QSplitter(QtCore.Qt.Horizontal, self.main_widget)
        splitter.addWidget(heatmap)
        frame.setLayout(userside)
        splitter.addWidget(frame)
        layout.addWidget(splitter)

        # fill the values for the gui
        # add the sample field combobox values
        for cfield in exp.sample_metadata.columns:
            self.w_field.addItem(cfield)

        heatmap.setFocusPolicy(QtCore.Qt.ClickFocus)
        heatmap.setFocus()

        self.plotaxes = heatmap.axes
        self.plotfigure = heatmap.figure

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

