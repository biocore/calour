from logging import getLogger

import ipywidgets
from IPython.display import display, clear_output, Javascript
import matplotlib

from .plotgui import PlotGUI
from .._doc import ds


logger = getLogger(__name__)


@ds.with_indent(4)
class PlotGUI_Jupyter(PlotGUI):
    '''Jupyter GUI of plotting.

    Attributes
    ----------

    Parameters
    ----------
    %(PlotGUI.parameters)s
    '''

    @ds.with_indent(8)
    def __init__(self, **kwargs):
        '''Init the GUI using the interactive Jupyter Notebook framework.

        .. note:: in order to use the interactive features, you need to first use
        the ``%matplotlib notebook`` directive inside the notebook

        Keyword Arguments
        -----------------
        %(PlotGUI.parameters)s
        '''
        super().__init__(**kwargs)
        # create the figure to plot the heatmap into
        self._set_figure(None, kwargs['tree_size'])
        if matplotlib.get_backend() != 'nbAgg':
            logger.warning('You need to set up jupyter notebook with `%matplotlib notebook`')

    def __call__(self):
        super().__call__()
        layout = ipywidgets.Layout(width='100%', word_wrap='normal', white_space='nowrap')
        self._ipyw_sid = ipywidgets.Text(
            value='-', placeholder='Sample ID', description='Sample ID', layout=layout)
        self._ipyw_fid = ipywidgets.Text(
            value='-', placeholder='Feature ID', description='Feature ID', layout=layout)
        self._ipyw_abund = ipywidgets.FloatText(
            value=0, placeholder='Abundance', description='Abundance', layout=layout)
        self._ipyw_selected = ipywidgets.Label('0 features are selected')
        # display selected samples/features
        display(self._ipyw_selected)
        display(ipywidgets.HBox([self._ipyw_sid, self._ipyw_fid, self._ipyw_abund]))

        self._ipyw_scol = ipywidgets.Dropdown(
            options=self.exp.sample_metadata.columns.tolist(),
            layout=ipywidgets.Layout(width='20%'))
        self._ipyw_scol.observe(self._on_change(axis=0))
        self._ipyw_fcol = ipywidgets.Dropdown(
            options=self.exp.feature_metadata.columns.tolist(),
            layout=ipywidgets.Layout(width='20%'))
        self._ipyw_fcol.observe(self._on_change(axis=1))
        self._ipyw_smeta = ipywidgets.Text(
            '-', placeholder='sample meta', description='',
            layout=ipywidgets.Layout(width='80%'))
        self._ipyw_fmeta = ipywidgets.Text(
            '-', placeholder='feature meta', description='',
            layout=ipywidgets.Layout(width='80%'))

        display(ipywidgets.HBox([self._ipyw_scol, self._ipyw_smeta]))
        display(ipywidgets.HBox([self._ipyw_fcol, self._ipyw_fmeta]))

        # # display zoom buttons
        # zoom_in_y = ipywidgets.Button(description='+', width='3%')
        # zoom_in_y.on_click(self._zoom_in_y)
        # zoom_out_y = ipywidgets.Button(description='-', width='3%')
        # zoom_out_y.on_click(self._zoom_out_y)
        # display(ipywidgets.HBox([zoom_in_y, zoom_out_y]))

        print_axes_lim = ipywidgets.Button(
            description='print axes ranges',
            tooltip=('Print the x axis and y axis ranges. Useful to pass to `rect` '
                     'parameter of heatmap() to zoom in the region automatically.'))
        print_axes_lim.on_click(self._print_axes_lim)
        # TODO
        save_selection = ipywidgets.Button(
            description='Save',
            tooltip='Save the selection(s)')
        save_selection.on_click(self._save)

        annotate_selection = ipywidgets.Button(description='Annotate')
        annotate_selection.on_click(self._annotate)
        display(ipywidgets.HBox([print_axes_lim, save_selection, annotate_selection]))

        # display annotation for the selection
        self._ipyw_annt = ipywidgets.HTML(
            'no annotation found',
            layout=ipywidgets.Layout(height='100px', overflow_y='auto'))
        display(self._ipyw_annt)
        # display(self.figure)

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
        ax.set_ylim(ylim_lower,
                    ylim_lower + (ylim_upper - ylim_lower) * self.zoom_scale)
        self.figure.canvas.draw()
        clear_output(wait=True)
        display(self.figure)

    def _print_axes_lim(self, button):
        ax = self.ax_hm
        ylim_lower, ylim_upper = ax.get_ylim()
        xlim_lower, xlim_upper = ax.get_xlim()
        print([xlim_lower, xlim_upper, ylim_lower, ylim_upper])

    def _annotate(self, button):
        '''Add annotation of the selected features to the database. '''
        if self._annotation_db is None:
            logger.warning('No database with add annotation capability selected (use plot(...,databases=[dbname])')
            return

        # get the sequences of the selection
        seqs = []
        for cseqpos in self.selected_features.keys():
            seqs.append(self.exp.feature_metadata.index[cseqpos])

        # from calour.annotation import annotate_bacteria_gui
        self._annotation_db.add_annotation(seqs, self.exp)

    def _popup(self, body, title="Calour"):
        data = """
                require(
                ["base/js/dialog"],
                function(dialog) {
                    dialog.modal({
                        title: '%s',
                        body: '%s',
                        buttons: {
                            'OK': {}
                        }
                    });
                });""" % (title, body)
        display(Javascript(data))

    def _save_seqs(self, name, button):
        # if no name entered, just warn
        if name == '':
            self._popup('No name supplied. Sequences not saved')
            return

        # get the sequences of the selection
        seqs = []
        for cseqpos in self.selected_features.keys():
            seqs.append(self.exp.feature_metadata.index[cseqpos])

        self.exp.feature_metadata[name] = False
        self.exp.feature_metadata.loc[seqs, name] = True
        button.description = 'Saved'
        self._popup('Saved. Use newexp=exp.filter_by_metadata("%s", [True],axis="f") to obtain filtered experiment' % name)

    def _save(self, button):
        '''Save feature collection to the experiment

        opens a textbox and save button. when save button is clicked, a column (with the name entered in the textbox)
        is added to the experiment feature_metadata, containing True on selected features, false on others

        In order to get a new experiment with only the saved features, you can use:

        newexp = exp.filter_by_metadata(SAVE_NAME, [True], axis='f')
         '''
        a = ipywidgets.Text()
        b = ipywidgets.Button(
            description='Save selection:',
            tooltip='Save the selection(s)')
        b.on_click(lambda x: self._save_seqs(a.value, x))
        ui = ipywidgets.HBox([a, b])
        display(ui)

    def show_info(self):
        sid, fid, abd, annt = self.get_info()
        self._ipyw_sid.value = str(sid)
        self._ipyw_fid.value = str(fid)
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
                annt_type = details.get('annotationtype', 'None')
                # annt_id = details.get('annotationid', 'NA')
                ccolor = colors.get(annt_type, 'black')
                # if we have annotations, add them
                if '_db_interface' in details:
                    link_address = details['_db_interface'].get_annotation_website(details)
                    cline = ('<p><a href="%s" target="_blank"><span style="color:%s;">%s</span></a></p>' % (link_address, ccolor, cstr))
                # otherwise - no annotations from the database
                else:
                    cline = ('<p><span style="color:%s;">%s</span></p>' % (ccolor, cstr))
                idata.append(cline)
        except Exception as e:
            # use try/except to catch and show the error; otherwise the error goes unnoticed
            self._ipyw_annt.value = repr(e)
        else:
            self._ipyw_annt.value = ''.join(idata)
