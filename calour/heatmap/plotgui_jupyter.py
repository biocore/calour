from logging import getLogger
import ipywidgets
from IPython.display import display, clear_output

from .plotgui import PlotGUI


logger = getLogger(__name__)


class PlotGUI_Jupyter(PlotGUI):
    '''Jupyter GUI of plotting.

    Attributes
    ----------

    Parameters
    ----------
    '''
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

    def __call__(self):
        super().__call__()
        # self.labtax = Label('Feature:-')
        self._labtax = ipywidgets.Label('-')
        # self.labsamp = Label('Sample:')
        self._labsamp = ipywidgets.Label('-')
        self._copy_sample_info = ipywidgets.Button(description='CP', width='2%')
        self._select_sample_info = ipywidgets.Dropdown(
            options=self.exp.sample_metadata.columns.tolist(),
            width='10%', max_width='10%', overflow_x='auto')
        self._copy_feature_info = ipywidgets.Button(description='CP', width='2%')
        self._select_feature_info = ipywidgets.Dropdown(
            options=self.exp.feature_metadata.columns.tolist(), width='10%')

        self._lababund = ipywidgets.Label('abundance: -')
        self._copy_view = ipywidgets.Button(description='Copy View')
        self._copy_view.on_click(self._copy_view)

        self._lab_selected = ipywidgets.Label('Selected: 0')
        self._save_selection = ipywidgets.Button(description='Save')
        self._annotate_selection = ipywidgets.Button(description='Annotate')
        self._annotate_selection.on_click(lambda f: _annotate(f, self))

        self._labdb = ipywidgets.HTML('?')
        self._labdb.layout.overflow = 'auto'
        self._labdb.layout.overflow_x = 'auto'
        self._labdb.layout.max_height = '50px'
        self._labdb.layout.white_space = 'nowrap'
        self._labdb.layout.border = '5px solid gray;'
        self._labdb.background_color = 'red'
        # self.labdb.layout.width = '200px'
        self._zoomin = ipywidgets.Button(description='+', width='3%')
        self._zoomin.on_click(self._zoom_in)
        self._zoomout = ipywidgets.Button(description='-', width='3%')
        self._zoomout.on_click(self._zoom_out)
        # display(HBox([self._zoomin,self._zoomout]))

        display(ipywidgets.HBox([self._copy_feature_info, self._select_feature_info, self._labtax]))
        display(ipywidgets.HBox([self._copy_sample_info, self._select_sample_info, self._labsamp]))
        display(ipywidgets.HBox([self._lababund, self._copy_view]))
        display(ipywidgets.HBox([self._lab_selected, self._save_selection, self._annotate_selection]))
        display(self._labdb)
    @staticmethod
    def _zoom_in(b):
        ax = self.figure.gca()
        ylim_lower, ylim_upper = ax.get_ylim()
        xlim_lower, xlim_upper = ax.get_xlim()
        ax.set_ylim(
            ylim_lower,
            ylim_lower + (ylim_upper - ylim_lower) / self.zoom_scale)
        self.figure.canvas.draw()
        clear_output(wait=True)
        display(self.figure)

    def _zoom_out(self, b):
        ax = self.figure.gca()
        ylim_lower, ylim_upper = ax.get_ylim()
        xlim_lower, xlim_upper = ax.get_xlim()
        ax.set_ylim(
                ylim_lower,
                ylim_lower + (ylim_upper - ylim_lower) * self.zoom_scale)
        self.figure.canvas.draw()
        clear_output(wait=True)
        display(self.figure)

    def show_info(self):
        sid, fid, abd, tax, annt = self.get_info()
        self._labtax.value = repr(tax)
        self._labsamp.value = sid
        self._labfeat.value = fid
        self._lababund.value = 'abundance: {:.01f}'.format(abd)
        self._lab_selected.value = '# selected: %d' % len(self.selected_features)

        idata = []
        colors = {'diffexp': 'blue',
                  'contamination': 'red',
                  'common': 'green',
                  'highfreq': 'green'}
        for cannt in annt:
            cstr = cannt[1]
            cannotationid = cannt[0]['annotationid']
            ccolor = colors.get(cannt[0], 'black')
            l = ('<style> a:link {color:%s; background-color:transparent; text-decoration:none}'
                 'a:visited {color:%s; background-color:transparent; text-decoration:none}</style>'
                 '<p style="color:%s;white-space:nowrap;">'
                 '<a href="http://amnonim.webfactional.com/scdb_website/annotation_info/%d"'
                 '   target="_blank">%s</a></p>') % (ccolor, ccolor, ccolor, cannotationid, cstr)
            idata.append(l)
        self._labdb.value = ''.join(idata)


def _copy_view(self, b):
    ax = self.figure.gca()
    ylim_lower, ylim_upper = ax.get_ylim()
    xlim_lower, xlim_upper = ax.get_xlim()
    print([xlim_lower, xlim_upper, ylim_lower, ylim_upper])


def _annotate(self, b):
    '''Add database annotation to selected features
    '''
    if self._annotation_db is None:
        logger.warn('No database with add annotation capability selected (use plot(...,databases=[dbname])')
        return

    # get the sequences of the selection
    seqs = []
    for cseqpos in self.selected_features.keys():
        seqs.append(self.exp.feature_metadata.index[cseqpos])

    # from calour.annotation import annotate_bacteria_gui
    self._annotation_db.add_annotation(seqs, self.exp)
