import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import Range1d
from math import pi


def block_plot(df=pd.DataFrame(), marker_name='marker', bubble_name='bubble'):
    """
    Function returning block-plot
    :param df: dataframe with data to be visualized
    :param marker_name: name of selected marker
    :param bubble_name: name of selected cell population
    :return: figure, the block-plot graph
    """
    if not df.empty:
        cats = df.index.tolist()

        val_1 = df.columns[0]
        val_2 = df.columns[1]
        kw = dict()
        kw['title'] = "Block plot: marker '%s' in population '%s'" % (marker_name, bubble_name)

        p = figure(background_fill_color="#efefef", x_range=cats, toolbar_location='above',
                   tools='pan, box_zoom,reset, wheel_zoom, save', height=400, width=1200,
                   **kw)

        start = df[val_1].tolist()
        end = df[val_2].tolist()
        p.vbar(cats, 0.2, start, end, fill_color='white', line_color='black')

        text_1 = ["1"] * len(start)
        text_2 = ["2"] * len(end)
        p.text(cats, start, text=text_1)
        p.text(cats, end, text=text_2)

        p.ygrid.minor_grid_line_color = 'navy'
        p.ygrid.minor_grid_line_alpha = 0.1
        p.yaxis.axis_label = "marker '%s' in population '%s'" % (marker_name, bubble_name)
        p.axis.major_label_text_font_size = "10pt"
        p.xaxis.major_label_orientation = pi / 2
        vals = df[val_1].tolist() + df[val_2].tolist()
        mi, ma = np.min(vals), np.max(vals)
        p.y_range = Range1d(mi - 0.5 * mi, ma + 0.5 * ma)
    else:
        p = figure(tools="", background_fill_color="#2F2F2F", x_range=['a', 'b', 'c'], toolbar_location=None,
                   height=400, width=1200)
    return p
