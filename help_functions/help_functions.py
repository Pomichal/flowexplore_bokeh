import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from functools import reduce
from bokeh.layouts import row, widgetbox, column
from functools import partial
from bokeh.models import Select
from bokeh.models.widgets import Button, Panel, PreText, DateRangeSlider, RangeSlider, CheckboxGroup


def prepare_data(df_patient, df_coordinates):
    """
    creates a dataframe with patient data and [x,y] coordinates for each cluster.

    :param df_patient: dataframe with patients data for each cluster
    :param df_coordinates: dataframe with coordinates for clusters
    """
    # print("coor", df_coordinates.iloc[:, 1].values)
    df_patient['x'] = df_coordinates.iloc[:, 1].values
    df_patient['y'] = df_coordinates.iloc[:, 2].values
    df_patient['populationID'] = -1
    data = df_patient.drop('Unnamed: 0', axis=1)
    # print(df_patient.head())

    # rename attributes example
    # data.rename(index=str, columns={"csv.pbm_038.count": "count"}, inplace=True)
    return data


def create_color_map():

    c_dict = {'red': ((0.0, 0.0, 0.0),
                      (0.25, 0.0, 0.0),
                      (0.5, 0.8, 1.0),
                      (0.75, 1.0, 1.0),
                      (1.0, 0.4, 1.0)),

              'green': ((0.0, 0.0, 0.0),
                        (0.25, 0.0, 0.0),
                        (0.5, 0.9, 0.9),
                        (0.75, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),
              'blue': ((0.0, 0.0, 0.4),
                       (0.25, 1.0, 1.0),
                       (0.5, 1.0, 0.8),
                       (0.75, 0.0, 0.0),
                       (1.0, 0.0, 0.0))
              }

    blue_red1 = LinearSegmentedColormap('BlueRed1', c_dict)

    colors = []
    for c in range(0, 256):
        rgb = blue_red1(c)[:3]  # will return rgba, we take only first 3 so we get rgb
        colors.append(mpl.colors.rgb2hex(rgb))

    return colors


def scale(old_value, old_min, old_max, new_min=25, new_max=50):
    """
    Scale data into range new_min, new_max
    :param old_value:
    :param old_min:
    :param old_max:
    :param new_min:
    :param new_max:
    :return: scaled value
    """
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    return (((old_value - old_min) * new_range) / old_range) + new_min


def create_stats_tables(pops, pats_data, viz_df):  # TODO remove error, if running without coordinates data
    b = pops['population_name'].values
    m = reduce(lambda s, q: s | q, map(lambda p: set(pats_data[p].columns.values), pats_data.keys()))
    iterables = [b, m]

    df_stats = pd.DataFrame(index=pd.MultiIndex.from_product(iterables), columns=pats_data.keys())

    cell_count_column = 'None'

    for pat in pats_data:
        df_patient = pats_data[pat]

        # select column with cell count
        # TODO better condition to filter columns and find cell count
        int_cols = list(filter(
            lambda a: (not "id" in a.lower()) and df_patient[a].nunique() > 2 and
                      ('int' in str(df_patient[a].values.dtype)), df_patient.columns))

        if len(int_cols) != 1:  # couldn't find the cell count column
            return df_stats, m
        else:
            cell_count_column = int_cols[0]
            cell_sum = df_patient[cell_count_column].sum()
            for idx_b, bu in enumerate(b):
                clusters = df_patient[viz_df['populationID'] == idx_b]
                for idx_m, ma in enumerate(m):
                    if ma != cell_count_column and ma in clusters.columns:
                        values = map(lambda a, count: a * clusters.loc[count, cell_count_column],
                                     clusters[ma].dropna(), clusters[ma].index.values)
                        df_stats.loc[(bu, ma), pat] = reduce(lambda p, q: p + q, list(values), 0) / cell_sum
                    else:
                        df_stats.loc[(bu, ma), pat] = reduce(lambda p, q: p + q, clusters[cell_count_column].values)
    df_stats = df_stats.astype(float)
    return df_stats, m


def find_healthy(pats_data, c_data):
    measurements = []
    for pat in pats_data.keys():
        pat_name = "-".join(pat.split("-")[:2])
        if pat_name not in c_data.index:
            measurements += [pat]

    new_df = pd.DataFrame(columns=['measurements', 'patient'])

    for m in measurements:
        df = pd.DataFrame([[m, 'healthy']], columns=['measurements', 'patient'])
        new_df = new_df.append(df)
    return new_df.reset_index(drop=True)

# trying drawing using graphs, but missing easily moving of vertices
# def create_figure2(df):
#     N = len(df)
#     node_indices = list(range(1, N + 1))
#
#     plot = figure(title='Graph Layout Demonstration', x_range=(-1.1, 600), y_range=(-1.1, 600),
#                   tools='pan, wheel_zoom, box_select', toolbar_location='above')
#
#     graph = GraphRenderer()
#
#     graph.node_renderer.data_source.add(node_indices, 'index')
#     graph.node_renderer.data_source.add(Spectral8, 'color')
    # graph.node_renderer.glyph = Circle(radius=15)
    #
    # graph.selection_policy = NodesAndLinkedEdges()
    #
    # graph.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=1)
    # graph.edge_renderer.data_source.data = dict(
    #     start=edges['edges.from'].tolist(),
    #     end=edges['edges.to'].tolist())
    #
    # start of layout code
    # xx = df['x']
    # yy = df['y']
    # graph_layout = dict(zip(node_indices, zip(xx, yy)))
    # graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
    #
    # plot.renderers.append(graph)
    #
    # draw_tool = PointDrawTool(add=False, renderers=[graph])
    # plot.add_tools(draw_tool)
    # return plot
