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
        layout = ipywidgets.Layout(width='100%')
        self._ipyw_sid = ipywidgets.Text(
            value='-', placeholder='Sample ID', description='Sample ID', layout=layout)
        self._ipyw_fid = ipywidgets.Text(
            value='-', placeholder='Feature ID', description='Feature ID', layout=layout)
        self._ipyw_abund = ipywidgets.FloatText(
            value=0, placeholder='Abundance', description='Abundance', layout=layout)
        self._ipyw_selected = ipywidgets.Label('0 features are selected')
        # display selected samples/features
        display(ipywidgets.VBox(
            [self._ipyw_selected, self._ipyw_sid, self._ipyw_fid, self._ipyw_abund]))

        self._ipyw_scol = ipywidgets.Dropdown(
            options=self.exp.sample_metadata.columns.tolist(), width='10%')
        self._ipyw_scol.observe(self._on_change(axis=0))
        self._ipyw_fcol = ipywidgets.Dropdown(
            options=self.exp.feature_metadata.columns.tolist(), width='10%')
        self._ipyw_fcol.observe(self._on_change(axis=1))
        self._ipyw_smeta = ipywidgets.Text(
            '-', placeholder='sample meta', description='', width='100%')
        self._ipyw_fmeta = ipywidgets.Text(
            '-', placeholder='feature meta', description='', width='100%')

        display(ipywidgets.HBox([self._ipyw_scol, self._ipyw_smeta]))
        display(ipywidgets.HBox([self._ipyw_fcol, self._ipyw_fmeta]))

        # # display zoom buttons
        # zoom_in_y = ipywidgets.Button(description='+', width='3%')
        # zoom_in_y.on_click(self._zoom_in_y)
        # zoom_out_y = ipywidgets.Button(description='-', width='3%')
        # zoom_out_y.on_click(self._zoom_out_y)
        # display(ipywidgets.HBox([zoom_in_y, zoom_out_y]))

        print_axes_lim = ipywidgets.Button(description='print axes ranges')
        print_axes_lim.on_click(self._print_axes_lim)
        # TODO
        save_selection = ipywidgets.Button(description='Save')
        save_selection.on_click(self._annotate)
        annotate_selection = ipywidgets.Button(description='Annotate')
        annotate_selection.on_click(self._annotate)
        display(ipywidgets.HBox([print_axes_lim, save_selection, annotate_selection]))

        # display annotation for the selection
        self._ipyw_annt = ipywidgets.HTML('no annotation found')
        # self._ipyw_annt.layout.overflow = 'auto'
        # self._ipyw_annt.layout.overflow_x = 'auto'
        # self._ipyw_annt.layout.max_height = '50px'
        # self._ipyw_annt.layout.white_space = 'nowrap'
        # self._ipyw_annt.layout.border = '5px solid gray;'
        # self._ipyw_annt.background_color = 'red'
        # self.ipywdb.layout.width = '200px'
        display(self._ipyw_annt)

    def _on_change(self, axis=0):
        '''Upon change in the dropdown sample or feature widgets, update their
        metadata values.'''
        def inner(change):
            if change['type'] == 'change' and change['name'] == 'value':
                col = change['new']
                if axis == 0:
                    # need to convert all other types to str because it is a text widget.
                    self._ipyw_smeta.value = str(
                        self.exp.sample_metadata[col].iloc[self.current_select[axis]])
                elif axis == 1:
                    self._ipyw_fmeta.value = str(
                        self.exp.feature_metadata[col].iloc[self.current_select[axis]])
        return inner

    def _zoom_in_y(self, button):
        ax = self.figure.gca()
        ylim_lower, ylim_upper = ax.get_ylim()
        xlim_lower, xlim_upper = ax.get_xlim()
        ax.set_ylim(
            ylim_lower,
            ylim_lower + (ylim_upper - ylim_lower) / self.zoom_scale)
        self.figure.canvas.draw()
        clear_output(wait=True)
        display(self.figure)

    def _zoom_out_y(self, button):
        ax = self.figure.gca()
        ylim_lower, ylim_upper = ax.get_ylim()
        xlim_lower, xlim_upper = ax.get_xlim()
        ax.set_ylim(
                ylim_lower,
                ylim_lower + (ylim_upper - ylim_lower) * self.zoom_scale)
        self.figure.canvas.draw()
        clear_output(wait=True)
        display(self.figure)

    def _print_axes_lim(self, button):
        ax = self.figure.gca()
        ylim_lower, ylim_upper = ax.get_ylim()
        xlim_lower, xlim_upper = ax.get_xlim()
        print([xlim_lower, xlim_upper, ylim_lower, ylim_upper])

    def _annotate(self, button):
        '''Add annotation of the selected features to the database. '''
        if self._annotation_db is None:
            logger.warn('No database with add annotation capability selected (use plot(...,databases=[dbname])')
            return

        # get the sequences of the selection
        seqs = []
        for cseqpos in self.selected_features.keys():
            seqs.append(self.exp.feature_metadata.index[cseqpos])

        # from calour.annotation import annotate_bacteria_gui
        self._annotation_db.add_annotation(seqs, self.exp)

    def show_info(self):
        sid, fid, abd, tax, annt = self.get_info()
        self._ipyw_sid.value = sid
        self._ipyw_fid.value = fid
        self._ipyw_abund.value = abd
        self._ipyw_selected.value = '%d features are selected' % len(self.selected_features)
        # need to convert all other types to str because it is a text widget.
        self._ipyw_smeta.value = str(self.exp.sample_metadata.loc[sid, self._ipyw_scol.value])
        self._ipyw_fmeta.value = str(self.exp.feature_metadata.loc[fid, self._ipyw_fcol.value])

        idata = []
        colors = {'diffexp': 'blue',
                  'contamination': 'red',
                  'common': 'green',
                  'highfreq': 'green'}

        try:
            for cannt in annt:
                cstr = cannt[1]
                details = cannt[0]
                print(cannt)
                annt_type = details.get('annotationtype', 'None')
                annt_id = details['annotationid']
                ccolor = colors.get(annt_type, 'black')
                l = ('<style> a:link {color:%s; background-color:transparent; text-decoration:none}'
                     'a:visited {color:%s; background-color:transparent; text-decoration:none}</style>'
                     '<p style="color:%s;white-space:nowrap;">'
                     '<a href="http://dbbact.org/annotation_info/%d"'
                     '   target="_blank">%s</a></p>') % (ccolor, ccolor, ccolor, annt_id, cstr)
                idata.append(l)
        except Exception as e:
            # use try/except to catch and show the error; otherwise the error goes unnoticed
            self._ipyw_annt.value = repr(e)
        else:
            self._ipyw_annt.value = ''.join(idata)
