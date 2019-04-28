import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import Span, Range1d
import random
from math import pi


def generate_random_color():
    return '#%02X%02X%02X' % (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def diff_plot(df_diff=pd.DataFrame(), marker_name='marker', bubble_name='bubble'):
    if not df_diff.empty:
        cats = df_diff.index.tolist()

        value = df_diff.columns[0]

        zero_diff = Span(location=0, dimension='width', line_color='red', line_dash='dashed', line_width=3)

        kw = dict()
        kw['title'] = "Difference from reference group: marker '%s' in population '%s'" % (marker_name, bubble_name)

        p = figure(background_fill_color="#efefef", x_range=cats, toolbar_location='above',
                   tools='pan, box_zoom,reset, wheel_zoom, save', height=400,
                   **kw)
        p.add_layout(zero_diff)

        start = [0] * len(cats)
        end = df_diff[value].tolist()
        colors = [generate_random_color() for _ in range(0, len(df_diff))]
        # print(colors)
        p.vbar(cats, 0.2, start, end, fill_color=colors, line_color='black')
        # p.vbar(cats, 0.4, q1[value], q2[value], fill_color="#3B8686", line_color="black")
        p.ygrid.minor_grid_line_color = 'navy'
        p.ygrid.minor_grid_line_alpha = 0.1
        p.yaxis.axis_label = "marker '%s' in population '%s'" % (marker_name, bubble_name)
        p.axis.major_label_text_font_size = "10pt"
        p.xaxis.major_label_orientation = pi / 2
        mi, ma = np.min(end), np.max(end)
        # print("MIN, MAX", mi, ma)
        if mi + ma < 0:
            p.y_range = Range1d(mi + 0.5*mi, -mi - 0.5*mi)
        else:
            p.y_range = Range1d(-ma - 0.5*ma, ma + 0.5*ma)

    else:
        p = figure(tools="", background_fill_color="#efefef", x_range=['a', 'b', 'c'], toolbar_location=None,
                   height=400)

    return p

