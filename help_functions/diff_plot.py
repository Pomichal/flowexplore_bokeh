import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import Span, Range1d
import random
from math import pi


def calculate_diff(b, m, stats_df, group_names, groups_dict, mean_or_med=0):
    """
    calculate difference from reference group
    :param b: name of the population (bubble)
    :param m: name of the marker
    :param stats_df: dataframe with all measurements and marker values for every population
    :param group_names: names of patient groups
    :param groups_dict: dict with groups data
    :param mean_or_med: 0- mean, 1-median
    :return: dataframe with difference from reference group for every group
    """
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


def generate_random_color():
    return '#%02X%02X%02X' % (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def diff_plot(df_diff=pd.DataFrame(), marker_name='marker', bubble_name='bubble'):
    """
    shows the difference from reference group
    :param df_diff: data with difference from reference group
    :param marker_name: name of the selected marker
    :param bubble_name: name of the selected population (bubble)
    :return: figure (diff plot)
    """
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
        p.vbar(cats, 0.2, start, end, fill_color=colors, line_color='black')
        p.ygrid.minor_grid_line_color = 'navy'
        p.ygrid.minor_grid_line_alpha = 0.1
        p.yaxis.axis_label = "difference on marker '%s' in population '%s'" % (marker_name, bubble_name)
        p.axis.major_label_text_font_size = "10pt"
        p.xaxis.major_label_orientation = pi / 2
        mi, ma = np.min(end), np.max(end)
        if mi + ma < 0:
            p.y_range = Range1d(mi + 0.5*mi, -mi - 0.5*mi)
        else:
            p.y_range = Range1d(-ma - 0.5*ma, ma + 0.5*ma)

    else:
        p = figure(tools="", background_fill_color="#efefef", x_range=['a', 'b', 'c'], toolbar_location=None,
                   height=400)

    return p

