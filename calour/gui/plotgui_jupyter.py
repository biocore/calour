import ipywidgets
from IPython.display import display, clear_output

from calour.bactdb import BactDB
from calour.gui.plotgui import PlotGUI


class PlotGUI_Jupyter(PlotGUI):
    '''QT5 version of plot winfow GUI

    We open the figure as a widget inside the qt5 window
    '''
    def __init__(self, *kargs, **kwargs):
        PlotGUI.__init__(self, *kargs, **kwargs)
        self.bactdb = BactDB()

    def get_figure(self, newfig=None):
        fig = PlotGUI.get_figure(self, newfig=newfig)
        # self.labtax = Label('Feature:-')
        self.labtax = ipywidgets.Label('-')
        # self.labsamp = Label('Sample:')
        self.labsamp = ipywidgets.Label('-')
        self.copy_sample_info = ipywidgets.Button(description='C', width='2%')
        self.select_sample_info = ipywidgets.Dropdown(options=list(self.exp.sample_metadata.columns), width='10%', max_width='10%', overflow_x='auto')
        self.copy_feature_info = ipywidgets.Button(description='C', width='2%')
        self.select_feature_info = ipywidgets.Dropdown(options=list(self.exp.feature_metadata.columns), width='10%')

        self.labreads = ipywidgets.Label('Reads:-')
        self.copy_view = ipywidgets.Button(description='Copy View')
        self.copy_view.on_click(lambda f: _copy_view(f, self))

        self.lab_selected = ipywidgets.Label('Selected: 0')
        self.save_selection = ipywidgets.Button(description='Save')
        self.annotate_selection = ipywidgets.Button(description='Annotate')
        self.annotate_selection.on_click(lambda f: _annotate(f, self))

        self.labdb = ipywidgets.HTML('?')
        self.labdb.layout.overflow = 'auto'
        self.labdb.layout.overflow_x = 'auto'
        self.labdb.layout.max_height = '50px'
        self.labdb.layout.white_space = 'nowrap'
        self.labdb.layout.border = '5px solid gray;'
        self.labdb.background_color = 'red'
        # self.labdb.layout.width = '200px'
        self.zoomin = ipywidgets.Button(description='+', width='3%')
        self.zoomin.on_click(lambda f: zoom_in(f, self))
        self.zoomout = ipywidgets.Button(description='-', width='3%')
        self.zoomout.on_click(lambda f: zoom_out(f, self))
        # display(HBox([self.zoomin,self.zoomout]))

        display(ipywidgets.HBox([self.copy_feature_info, self.select_feature_info, self.labtax]))
        # display(self.labtax)
        display(ipywidgets.HBox([self.copy_sample_info, self.select_sample_info, self.labsamp]))
        display(ipywidgets.HBox([self.labreads, self.copy_view]))
        # display(self.labreads)
        display(ipywidgets.HBox([self.lab_selected, self.save_selection, self.annotate_selection]))
        display(self.labdb)
        return fig

    def update_info(self):
        # taxname = self.exp.feature_metadata['taxonomy'][self.last_select_feature]
        taxname = self.exp.feature_metadata[self.select_feature_info.value][self.last_select_feature]
        # sampname = self.exp.sample_metadata.index[self.last_select_sample]
        sampname = self.exp.sample_metadata[self.select_sample_info.value][self.last_select_sample]
        sequence = self.exp.feature_metadata.index[self.last_select_feature]
        # self.labtax.value = 'Feature: %s' % taxname
        # self.labsamp.value = 'Sample: %s' % sampname
        self.labtax.value = str(taxname)
        self.labsamp.value = str(sampname)
        self.labreads.value = 'Reads:{:.01f}'.format(self.exp.get_data()[self.last_select_sample, self.last_select_feature])
        self.lab_selected.value = 'Selected: %d' % len(self.selected_features)
        info = self.bactdb.get_seq_annotation_strings(sequence)
        idata = ''
        for cinfo in info:
            cstr = cinfo[1]
            ccolor = self._get_color(cinfo[0])
            idata += '<style> a:link {color:%s; background-color:transparent; text-decoration:none} a:visited {color:%s; background-color:transparent;'
            ' text-decoration:none}</style>' % (ccolor, ccolor)
            idata += '<p style="color:%s;white-space:nowrap;"><a href="http://amnonim.webfactional.com/scdb_website/exp_info/19" target="_blank">%s</a></p>' % (ccolor, cstr)
        self.labdb.value = idata

    def _get_color(self, details):
        if details['annotationtype'] == 'diffexp':
            ccolor = 'blue'
        elif details['annotationtype'] == 'contamination':
            ccolor = 'red'
        elif details['annotationtype'] == 'common':
            ccolor = 'green'
        elif details['annotationtype'] == 'highfreq':
            ccolor = 'green'
        else:
            ccolor = 'black'
        return ccolor


def _copy_view(b, hdat):
    ax = hdat.fig.gca()
    ylim_lower, ylim_upper = ax.get_ylim()
    xlim_lower, xlim_upper = ax.get_xlim()
    print([xlim_lower, xlim_upper, ylim_lower, ylim_upper])


def zoom_in(b, hdat):
    ax = hdat.fig.gca()
    ylim_lower, ylim_upper = ax.get_ylim()
    xlim_lower, xlim_upper = ax.get_xlim()
    ax.set_ylim(
        ylim_lower,
        ylim_lower + (ylim_upper - ylim_lower) / hdat.zoom_scale)
    hdat.canvas.draw()
    clear_output(wait=True)
    display(hdat.fig)


def zoom_out(b, hdat):
    ax = hdat.fig.gca()
    ylim_lower, ylim_upper = ax.get_ylim()
    xlim_lower, xlim_upper = ax.get_xlim()
    ax.set_ylim(
            ylim_lower,
            ylim_lower + (ylim_upper - ylim_lower) * hdat.zoom_scale)
    hdat.canvas.draw()
    clear_output(wait=True)
    display(hdat.fig)


def _annotate(b, hdat):
    '''Add database annotation to selected features
    '''
    from calour.annotation import annotate_bacteria_gui

    # get the sequences of the selection
    seqs = []
    for cseqpos in hdat.selected_features.keys():
        seqs.append(hdat.exp.feature_metadata.index[cseqpos])

    annotate_bacteria_gui(seqs, hdat.exp)
