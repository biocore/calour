import importlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from logging import getLogger


logger = getLogger(__name__)


def class_for_name(module_name, class_name):
    # load the gui module
    try:
        m = importlib.import_module(module_name)
    except:
        raise ValueError('gui module %s not found' % module_name)
    # get the class
    try:
        c = getattr(m, class_name)
    except:
        raise ValueError('class %s not found in module %s. Is it a GUI module?' % class_name)
    return c


class PlotGUI:
    '''The base class for heatmap GUI

    Various gui classes inherit this class
    '''
    def __init__(self, exp, zoom_scale=2, scroll_offset=0):
        '''Init the gui windows class

        Store the experiment and the gui
        '''
        self.exp = exp
        self.zoom_scale = zoom_scale
        self.scroll_offset = scroll_offset

    def get_figure(self, newfig=None):
        ''' Get the figure to plot the heatmap into

        newfig: None or mpl.figure or False (optional)
            if None (default) create a new mpl figure.
            if mpl.figure use this figure to plot into.
            if False, use current figure to plot into.
        '''
        if newfig is None:
            fig = plt.figure()
        elif isinstance(newfig, mpl.figure.Figure):
            fig = newfig
        else:
            fig = plt.gcf()
        self.fig = fig
        return fig

    def connect_functions(self, fig):
        '''Connect to the matplotlib callbacks for key and mouse
        '''
        self.canvas = fig.canvas
        self.canvas.mpl_connect('scroll_event', lambda f: _scroll_callback(f, hdat=self))
        self.canvas.mpl_connect('key_press_event', lambda f: _key_press_callback(f, hdat=self))
        self.canvas.mpl_connect('button_press_event', lambda f: _button_press_callback(f, hdat=self))

    def update_info(self):
        '''Update info when a new feature/sample is selected
        '''
        pass


def _button_press_callback(event, hdat):
    rx = int(round(event.xdata))
    ry = int(round(event.ydata))
    hdat.select_feature = ry
    hdat.select_sample = rx
    hdat.update_info()


def _key_press_callback(event, hdat):
    ax = event.inaxes
    ylim_lower, ylim_upper = ax.get_ylim()
    xlim_lower, xlim_upper = ax.get_xlim()

    # set the scroll offset
    if hdat.scroll_offset > 0:
        x_offset = 1
        y_offset = 1
    else:
        x_offset = xlim_upper - xlim_lower
        y_offset = ylim_upper - ylim_lower

    if event.key == 'shift+up' or event.key == '=':
        ax.set_ylim(
            ylim_lower,
            ylim_lower + (ylim_upper - ylim_lower) / hdat.zoom_scale)
    elif event.key == 'shift+down' or event.key == '-':
        ax.set_ylim(
            ylim_lower,
            ylim_lower + (ylim_upper - ylim_lower) * hdat.zoom_scale)
    elif event.key == 'shift+right' or event.key == '+':
        ax.set_xlim(
            xlim_lower,
            xlim_lower + (xlim_upper - xlim_lower) / hdat.zoom_scale)
    elif event.key == 'shift+left' or event.key == '_':
        ax.set_xlim(
            xlim_lower,
            xlim_lower + (xlim_upper - xlim_lower) * hdat.zoom_scale)
    elif event.key == 'down':
        ax.set_ylim(ylim_lower - y_offset, ylim_upper - y_offset)
    elif event.key == 'up':
        ax.set_ylim(ylim_lower + y_offset, ylim_upper + y_offset)
    elif event.key == 'left':
        ax.set_xlim(xlim_lower - x_offset, xlim_upper - x_offset)
    elif event.key == 'right':
        ax.set_xlim(xlim_lower + x_offset, xlim_upper + x_offset)
    else:
        return

#    plt.tight_layout()
    hdat.canvas.draw()


def _scroll_callback(event, hdat):
    ax = event.inaxes
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    xdata = event.xdata  # get event x location
    ydata = event.ydata  # get event y location
    x_left = xdata - cur_xlim[0]
    x_right = cur_xlim[1] - xdata
    y_top = ydata - cur_ylim[0]
    y_bottom = cur_ylim[1] - ydata
    if event.button == 'up':
        scale_factor = 1. / hdat.zoom_scale
    elif event.button == 'down':
        scale_factor = hdat.zoom_scale
    else:
        # deal with something that should never happen
        scale_factor = 1
        print(event.button)
    # set new limits
    ax.set_xlim([xdata - x_left * scale_factor,
                 xdata + x_right * scale_factor])
    ax.set_ylim([ydata - y_top * scale_factor,
                 ydata + y_bottom * scale_factor])

    hdat.canvas.draw()  # force re-draw
