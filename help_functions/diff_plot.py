import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import Span, Range1d
import random
from math import pi


def calculate_diff(b, m, stats_df, group_names, groups_dict, mean_or_med=0):
    # stats_df = pd.DataFrame(stats.loc[(b, m), :])
    diff_df = pd.DataFrame(index=list([tab.title for tab in group_names]), columns=['diff'])

    reference_level = 0
    for g in groups_dict:
        measurements = g[1]['measurements'].tolist()
        if g[1]['patient'].values.tolist()[0] == 'healthy':
            if mean_or_med == 0:
                reference_level = np.mean(list([stats_df.loc[measurement, (b, m)] for measurement in measurements]))
            else:
                reference_level = np.median(list([stats_df.loc[measurement, (b, m)] for measurement in measurements]))
            break
    for idx, g in enumerate(groups_dict):
        measurements = g[1]['measurements'].tolist()
        if mean_or_med == 0:
            group_level = np.mean(list([stats_df.loc[measurement, (b, m)] for measurement in measurements]))
        else:
            group_level = np.median(list([stats_df.loc[measurement, (b, m)] for measurement in measurements]))
        diff_df.loc[group_names[idx].title, 'diff'] = group_level - reference_level

    return diff_df
    # layout3.children[2] = diff_plot.diff_plot(diff_df, m, b)


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

