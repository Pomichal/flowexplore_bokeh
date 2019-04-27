import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import Span, Range1d
import random


def generate_random_color():
    return '#%02X%02X%02X' % (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def diff_plot(df_diff=pd.DataFrame()):
    if not df_diff.empty:
        cats = df_diff.index.tolist()

        value = df_diff.columns[0]

        zero_diff = Span(location=0, dimension='width', line_color='red', line_dash='dashed', line_width=3)

        p = figure(tools="", background_fill_color="#efefef", x_range=cats, toolbar_location=None)
        p.add_layout(zero_diff)

        start = [0] * len(cats)
        end = df_diff[value].tolist()
        colors = [generate_random_color() for _ in range(0,len(df_diff))]
        # print(colors)
        p.vbar(cats, 0.2, start, end, fill_color=colors, line_color='black')
        # p.vbar(cats, 0.4, q1[value], q2[value], fill_color="#3B8686", line_color="black")
        p.ygrid.minor_grid_line_color = 'navy'
        p.ygrid.minor_grid_line_alpha = 0.1
        mi, ma = np.min(end), np.max(end)
        print(mi, ma)
        if ma == 0 or mi > (-1) * ma:
            p.y_range = Range1d(mi, -mi)
        else:
            p.y_range = Range1d(-ma, ma)


    else:
        p = figure(tools="", background_fill_color="#efefef", x_range=['a', 'b', 'c'], toolbar_location=None)

    return p

