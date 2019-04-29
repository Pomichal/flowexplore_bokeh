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


def rainbow_color_map():
    colors = ['#0000FF', '#0002FF', '#0005FF', '#0008FF', '#000BFF', '#000EFF',
              '#0011FF', '#0014FF', '#0017FF', '#001AFF', '#001DFF', '#0020FF',
              '#0023FF', '#0026FF', '#0029FF', '#002CFF', '#002FFF', '#0032FF',
              '#0035FF', '#0038FF', '#003BFF', '#003EFF', '#0041FF', '#0044FF',
              '#0047FF', '#004AFF', '#004DFF', '#0050FF', '#0053FF', '#0056FF',
              '#0059FF', '#005CFF', '#005FFF', '#0062FF', '#0065FF', '#0068FF',
              '#006BFF', '#006EFF', '#0071FF', '#0074FF', '#0077FF', '#007AFF',
              '#007DFF', '#0080FF', '#0083FF', '#0086FF', '#0089FF', '#008CFF',
              '#008FFF', '#0092FF', '#0095FF', '#0098FF', '#009BFF', '#009EFF',
              '#00A1FF', '#00A4FF', '#00A7FF', '#00AAFF', '#00ADFF', '#00B0FF',
              '#00B3FF', '#00B6FF', '#00B9FF', '#00BCFF', '#00BFFF', '#00C2FF',
              '#00C5FF', '#00C8FF', '#00CBFF', '#00CEFF', '#00D1FF', '#00D4FF',
              '#00D7FF', '#00DAFF', '#00DDFF', '#00E0FF', '#00E3FF', '#00E6FF',
              '#00E9FF', '#00ECFF', '#00EFFF', '#00F2FF', '#00F5FF', '#00F8FF',
              '#00FBFF', '#00FFFF', '#02FFFB', '#05FFF8', '#08FFF5', '#0BFFF2',
              '#0EFFEF', '#11FFEC', '#14FFE9', '#17FFE6', '#1AFFE3', '#1DFFE0',
              '#20FFDD', '#23FFDA', '#26FFD7', '#29FFD4', '#2CFFD1', '#2FFFCE',
              '#32FFCB', '#35FFC8', '#38FFC5', '#3BFFC2', '#3EFFBF', '#41FFBC',
              '#44FFB9', '#47FFB6', '#4AFFB3', '#4DFFB0', '#50FFAD', '#53FFAA',
              '#56FFA7', '#59FFA4', '#5CFFA1', '#5FFF9E', '#62FF9B', '#65FF98',
              '#68FF95', '#6BFF92', '#6EFF8F', '#71FF8C', '#74FF89', '#77FF86',
              '#7AFF83', '#7DFF80', '#80FF7D', '#83FF7A', '#86FF77', '#89FF74',
              '#8CFF71', '#8FFF6E', '#92FF6B', '#95FF68', '#98FF65', '#9BFF62',
              '#9EFF5F', '#A1FF5C', '#A4FF59', '#A7FF56', '#AAFF53', '#ADFF50',
              '#B0FF4D', '#B3FF4A', '#B6FF47', '#B9FF44', '#BCFF41', '#BFFF3E',
              '#C2FF3B', '#C5FF38', '#C8FF35', '#CBFF32', '#CEFF2F', '#D1FF2C',
              '#D4FF29', '#D7FF26', '#DAFF23', '#DDFF20', '#E0FF1D', '#E3FF1A',
              '#E6FF17', '#E9FF14', '#ECFF11', '#EFFF0E', '#F2FF0B', '#F5FF08',
              '#F8FF05', '#FBFF02', '#FFFF00', '#FFFB00', '#FFF800', '#FFF500',
              '#FFF200', '#FFEF00', '#FFEC00', '#FFE900', '#FFE600', '#FFE300',
              '#FFE000', '#FFDD00', '#FFDA00', '#FFD700', '#FFD400', '#FFD100',
              '#FFCE00', '#FFCB00', '#FFC800', '#FFC500', '#FFC200', '#FFBF00',
              '#FFBC00', '#FFB900', '#FFB600', '#FFB300', '#FFB000', '#FFAD00',
              '#FFAA00', '#FFA700', '#FFA400', '#FFA100', '#FF9E00', '#FF9B00',
              '#FF9800', '#FF9500', '#FF9200', '#FF8F00', '#FF8C00', '#FF8900',
              '#FF8600', '#FF8300', '#FF8000', '#FF7D00', '#FF7A00', '#FF7700',
              '#FF7400', '#FF7100', '#FF6E00', '#FF6B00', '#FF6800', '#FF6500',
              '#FF6200', '#FF5F00', '#FF5C00', '#FF5900', '#FF5600', '#FF5300',
              '#FF5000', '#FF4D00', '#FF4A00', '#FF4700', '#FF4400', '#FF4100',
              '#FF3E00', '#FF3B00', '#FF3800', '#FF3500', '#FF3200', '#FF2F00',
              '#FF2C00', '#FF2900', '#FF2600', '#FF2300', '#FF2000', '#FF1D00',
              '#FF1A00', '#FF1700', '#FF1400', '#FF1100', '#FF0E00', '#FF0B00',
              '#FF0800', '#FF0500', '#FF0200', '#FF0000']
    return colors


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
