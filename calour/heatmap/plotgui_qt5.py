import sys
from logging import getLogger

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QMainWindow, QHBoxLayout, QVBoxLayout,
                             QSizePolicy, QWidget, QPushButton,
                             QLabel, QListWidget, QSplitter, QFrame,
                             QComboBox, QScrollArea, QListWidgetItem,
                             QDialogButtonBox, QApplication)

from .plotgui import PlotGUI
from .. import analysis


logger = getLogger(__name__)


class PlotGUI_QT5(PlotGUI):
    '''QT5 version of plot winfow GUI

    We open the figure as a widget inside the qt5 window

    Attributes
    ----------
    figure
    app : QT5 App created
    app_window : Windows belonging to the QT5 App
    databases :
    '''
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        # create qt app
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
            logger.debug('Qt app created')
        self.app = app
        if not hasattr(app, 'references'):
            # store references to all the windows
            app.references = set()
        # create app window
        self.app_window = ApplicationWindow(self)
        app.references.add(self.app_window)
        self.app_window.setWindowTitle("Calour")
        self.figure = self.app_window.plotfigure

    def __call__(self):
        logger.debug('opening plot window')
        super().__call__()
        try:
            self.app_window.show()
            self.app.exec_()
        finally:
            # clean up when the qt app is closed
            if self.app_window in self.app.references:
                logger.debug('removing window from app window list')
                self.app.references.remove(self.app_window)
            else:
                logger.debug('window not in app window list. Not removed')

    def show_info(self):
        if 'taxonomy' in self.exp.feature_metadata:
            taxname = self.exp.feature_metadata['taxonomy'][self.current_select[1]]
        else:
            taxname = 'NA'
        sequence = self.exp.feature_metadata.index[self.current_select[1]]
        self.app_window.w_taxonomy.setText('%r' % taxname)
        self.app_window.w_reads.setText('reads:{:.01f}'.format(self.exp.get_data()[self.current_select[0], self.current_select[1]]))
        # self.app_window.w_dblist.addItem(taxname)
        csample_field = str(self.app_window.w_field.currentText())
        self.app_window.w_field_val.setText(str(self.exp.sample_metadata[csample_field][self.current_select[0]]))

        self.app_window.w_dblist.clear()
        info = []
        for cdatabase in self.databases:
            try:
                cinfo = cdatabase.get_seq_annotation_strings(sequence)
                if len(cinfo) == 0:
                    cinfo = [[{'annotationtype': 'not found'}, 'No annotation found in database %s' % cdatabase.get_name()]]
                else:
                    for cannotation in cinfo:
                        cannotation[0]['_db_interface'] = cdatabase
            except:
                cinfo = 'error connecting to db %s' % cdatabase.get_name()
            info.extend(cinfo)
        self._display_annotation_in_qlistwidget(info)

    def _display_annotation_in_qlistwidget(self, info):
        '''Add a line to the annotation list
        Does not erase previous lines

        Parameters
        ----------
        info : list of (dict, string)
            dict : contains the key 'annotationtype' and determines the annotation color
            also contains all other annotation data needed for right click menu/double click
            string : str
                The string to add to the list
        '''
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
            self.app_window.w_dblist.addItem(newitem)


class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class ApplicationWindow(QMainWindow):
    def __init__(self, gui):
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
        self.w_sequence = QPushButton(text='Copy Seq')
        lbox_buttons.addWidget(self.w_sequence)
        self.w_info = QPushButton(text='Info')
        lbox_buttons.addWidget(self.w_info)
        self.w_annotate = QPushButton(text='Annotate')
        lbox_buttons.addWidget(self.w_annotate)
        userside.addLayout(lbox_buttons)
        # db annotations list
        self.w_dblist = QListWidget()
        self.w_dblist.itemDoubleClicked.connect(self.double_click_annotation)
        userside.addWidget(self.w_dblist)

        lbox_buttons_bottom = QHBoxLayout()
        self.w_save_fasta = QPushButton(text='Save Seqs')
        lbox_buttons_bottom.addWidget(self.w_save_fasta)
        self.w_enrichment = QPushButton(text='Enrichment')
        lbox_buttons_bottom.addWidget(self.w_enrichment)
        userside.addLayout(lbox_buttons_bottom)

        layout = QHBoxLayout(self.main_widget)
        heatmap = MplCanvas(self.main_widget, width=5, height=4, dpi=100)
        frame = QFrame()
        splitter = QSplitter(QtCore.Qt.Horizontal, self.main_widget)
        splitter.addWidget(heatmap)
        frame.setLayout(userside)
        splitter.addWidget(frame)
        layout.addWidget(splitter)

        # fill the values for the gui
        # add the sample field combobox values
        for cfield in gui.exp.sample_metadata.columns:
            self.w_field.addItem(cfield)

        heatmap.setFocusPolicy(QtCore.Qt.ClickFocus)
        heatmap.setFocus()

        self.plotaxes = heatmap.axes
        self.plotfigure = heatmap.figure
        self.gui = gui

        # link events to gui
        self.w_annotate.clicked.connect(self.annotate)
        self.w_sequence.clicked.connect(self.copy_sequence)
        self.w_save_fasta.clicked.connect(self.save_fasta)
        self.w_enrichment.clicked.connect(self.enrichment)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def copy_sequence(self):
        '''Copy the sequence to the clipboard
        '''
        cseq = self.gui.exp.feature_metadata.index[self.gui.current_select[1]]
        clipboard = QApplication.clipboard()
        clipboard.setText(cseq)

    def save_fasta(self):
        seqs = self.gui.get_selected_seqs()
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, caption='Save selected seqs to fasta')
        self.gui.exp.save_fasta(str(filename), seqs)

    def enrichment(self):
        group1_seqs = self.gui.get_selected_seqs()
        allseqs = self.gui.exp.feature_metadata.index.values
        group2_seqs = list(set(allseqs).difference(set(group1_seqs)))

        logger.debug('Getting experiment annotations for %d features' % len(allseqs))
        for cdb in self.gui.databases:
            if not cdb.can_feature_terms():
                continue
            logger.debug('Database: %s' % cdb.get_name())
            feature_terms = cdb.get_feature_terms(allseqs, self.gui.exp)
            logger.debug('got %d terms' % len(feature_terms))
            res = analysis.relative_enrichment(self.gui.exp, group1_seqs, feature_terms)
            logger.debug('Got %d enriched terms' % len(res))
            if len(res) == 0:
                QtWidgets.QMessageBox.information(self, "No enriched terms found",
                                                  "No enriched annotations found when comparing\n%d selected sequences to %d "
                                                  "other sequences" % (len(group1_seqs), len(group2_seqs)))
                return
            listwin = SListWindow(listname='enriched ontology terms')
            for cres in res:
                if cres['group1'] > cres['group2']:
                    ccolor = 'blue'
                else:
                    ccolor = 'red'
                cname = cres['description']
                listwin.add_item('%s - %f (selected %f, other %f) ' % (cname, cres['pval'], cres['group1'], cres['group2']), color=ccolor)
            listwin.exec_()

    def double_click_annotation(self, item):
        '''Show database information about the double clicked item in the list.

        Call the appropriate database for displaying the info
        '''
        data = item.data(QtCore.Qt.UserRole)
        db = data.get('_db_interface', None)
        if db is None:
            return
        db.show_annotation_info(data)

    def annotate(self):
        '''Add database annotation to selected features
        '''
        # get the database used to add annotation
        if self.gui._annotation_db is None:
            logger.warn('No database with add annotation capability selected (use plot(...,databases=[dbname])')
            return
        # get the sequences of the selection
        seqs = self.gui.get_selected_seqs()
        # annotate
        self.gui._annotation_db.add_annotation(seqs, self.gui.exp)


class SListWindow(QtWidgets.QDialog):
    def __init__(self, listdata=[], listname=None):
        '''Create a list window with items in the list and the listname as specified

        Parameters
        ----------
        listdata: list of str (optional)
            the data to show in the list
        listname: str (optional)
            name to display above the list
        '''
        super().__init__()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        if listname is not None:
            self.setWindowTitle(listname)

        self.layout = QVBoxLayout(self)

        self.w_list = QListWidget()
        self.layout.addWidget(self.w_list)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok)
        buttonBox.accepted.connect(self.accept)
        self.layout.addWidget(buttonBox)

        for citem in listdata:
            self.w_list.addItem(citem)

        self.show()
        self.adjustSize()

    def add_item(self, text, color='black'):
        '''Add an item to the list

        Parameters
        ----------
        text : str
            the string to add
        color : str (optional)
            the color of the text to add
        '''
        item = QtWidgets.QListWidgetItem()
        item.setText(text)
        if color == 'black':
            ccolor = QtGui.QColor(0, 0, 0)
        elif color == 'red':
            ccolor = QtGui.QColor(155, 0, 0)
        elif color == 'blue':
            ccolor = QtGui.QColor(0, 0, 155)
        elif color == 'green':
            ccolor = QtGui.QColor(0, 155, 0)
        item.setForeground(ccolor)
        self.w_list.addItem(item)
