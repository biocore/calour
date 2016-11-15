class HeatmapInteractor(object):
    '''A heatmap editor
    '''
    def __init__(self, fig):
        canvas = fig.canvas
        canvas.mpl_connect('scroll_event', self.scroll_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)

    def button_press_callback(self, event):
        rx = int(round(event.xdata))
        ry = int(round(event.ydata))
        print("row: %d    column: %d" % (rx, ry))

    def key_press_callback(self, event, base_scale=2., offset=1):
        ax = event.inaxes
        ylim_lower, ylim_upper = ax.get_ylim()
        xlim_lower, xlim_upper = ax.get_xlim()
        if event.key == '[':
            ax.set_ylim(
                ylim_lower,
                ylim_lower + (ylim_upper - ylim_lower) / base_scale)
        elif event.key == ']':
            ax.set_ylim(
                ylim_lower,
                ylim_lower + (ylim_upper - ylim_lower) * base_scale)
        elif event.key == '{':
            ax.set_xlim(
                xlim_lower,
                xlim_lower + (xlim_upper - xlim_lower) / base_scale)
        elif event.key == '}':
            ax.set_xlim(
                xlim_lower,
                xlim_lower + (xlim_upper - xlim_lower) * base_scale)
        elif event.key == 'down':
            ax.set_ylim(ylim_lower - offset, ylim_upper - offset)
        elif event.key == 'up':
            ax.set_ylim(ylim_lower + offset, ylim_upper + offset)
        elif event.key == 'right':
            ax.set_xlim(xlim_lower - offset, xlim_upper - offset)
        elif event.key == 'left':
            ax.set_xlim(xlim_lower + offset, xlim_upper + offset)
        else:
            print('Unknown key: %s' % event.key)
            return

        # plt.tight_layout()
        plt.draw()

    def scroll_callback(self, event, base_scale=2.):
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
            scale_factor = 1. / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        # set new limits
        ax.set_xlim([xdata - x_left * scale_factor,
                     xdata + x_right * scale_factor])
        ax.set_ylim([ydata - y_top * scale_factor,
                     ydata + y_bottom * scale_factor])

        plt.draw()  # force re-draw


def transition_index(l):
    '''Return the transition index of the list.

    Parameters
    ----------
    l : list, 1-D array, pd.Series
        l should have method of len and [
    '''
    for i in range(1, len(l)):
        if l[i] != l[i-1]:
            yield i


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    im = ax.imshow([[1, 2], [3, 4]], aspect='auto')
    ax.axvline(x=1, color='black')
    p = HeatmapInteractor(fig)

    plt.show()
