import pandas as pd
import numpy as np
import math
from os.path import join, dirname
from bokeh.layouts import row, widgetbox, column
from bokeh.models import Select, ColorBar, ColumnDataSource, HoverTool, PointDrawTool, \
    CustomJS, LassoSelectTool, GraphRenderer, StaticLayoutProvider, Circle, MultiLine
from bokeh.models.widgets import Button, Dropdown, TextInput, DataTable, TableColumn, NumberFormatter
from bokeh.models.mappers import LinearColorMapper
from bokeh.models.graphs import NodesAndLinkedEdges
from bokeh.plotting import curdoc, figure
from functools import reduce

from io import StringIO
import base64

import upload as up
import help_functions as hf

file_source = ColumnDataSource({'file_contents': [], 'file_name': []})

patient_data = pd.DataFrame()
coordinates = pd.DataFrame()
edges = pd.DataFrame()

df_patients = pd.DataFrame()
populations = pd.DataFrame()
source = ColumnDataSource()

population_colors = pd.read_csv(join(dirname(__file__), 'data/colors.csv'))


def file_callback(attr, old, new):
    global patient_data
    global coordinates
    global edges
    global df_patients
    global source
    # print('filename:', file_source.data['file_name'])
    raw_contents = file_source.data['file_contents'][0]

    # remove the prefix that JS adds
    prefix, b64_contents = raw_contents.split(",", 1)
    file_contents = base64.b64decode(b64_contents)
    file_io = StringIO(bytes.decode(file_contents))
    df = pd.read_csv(file_io)
    # print("file contents:")
    # print(df)
    if dropdown.value == 'patient_data':
        patient_data = df
        dropdown.menu[0] = ("patient data ok", 'patient_data')
    elif dropdown.value == 'coordinates':
        coordinates = df
        dropdown.menu[1] = ("coordinates ok", 'coordinates')
    elif dropdown.value == 'edges':
        edges = df
        dropdown.menu[2] = ("edges ok", 'edges')
    else:
        print("something went wrong, unknown dropdown value")
    if reduce(lambda a, b: a and b, [True if 'ok' in string[0] else False for string in dropdown.menu]):
        dropdown.button_type = "success"
        df_patients = hf.prepare_data(patient_data, coordinates)
        source = ColumnDataSource(df_patients)
        layout.children[1] = create_figure(df_patients)
        x.options = df_patients.columns.tolist()
        x.value = 'x'
        y.options = df_patients.columns.tolist()
        y.value = 'y'
        color.options = ['None'] + df_patients.columns.tolist()
        color.value = 'None'
        size.options = ['None'] + df_patients.columns.tolist()
        size.value = 'None'
        # print(df_patients.head())
        # print(edges.head())


def create_figure(df):
    if not df.empty:

        pop_names = [populations.iloc[pop_id]['population_name'] if pop_id != -1 else '???'
                     for pop_id in df['populationID']]
        source.add(pop_names, name='pop_names')

        x_title = x.value.title()
        y_title = y.value.title()

        kw = dict()
        kw['title'] = "%s vs %s" % (x_title, y_title)

        p = figure(plot_height=900, plot_width=1200,
                   tools='pan, box_zoom,reset, wheel_zoom, box_select, tap, save',
                   toolbar_location="above", **kw)
        p.add_tools(LassoSelectTool(select_every_mousemove=False))

        p.xaxis.axis_label = x_title
        p.yaxis.axis_label = y_title

        # mark populations
        line_color = [populations.iloc[pop_id]['color'] if pop_id != -1 else 'white'
                      for pop_id in df['populationID']]
        source.add(line_color, name='lc')

        line_width = [5 if lc != 'white' else 1 for lc in line_color]
        source.add(line_width, name='lw')

        if size.value != 'None':
            sizes = [hf.scale(value, df[size.value].min(),
                              df[size.value].max()) if not np.isnan(value) else 3 for value in df[size.value]]
        else:
            sizes = [15 for _ in df[x.value]]
        source.add(sizes, name='sz')

        if color.value != 'None':
            mapper = LinearColorMapper(palette=hf.create_color_map(),
                                       high=df[color.value].max(),
                                       high_color='red',
                                       low=df[color.value].min(),
                                       low_color='blue'
                                       )
            color_bar = ColorBar(color_mapper=mapper, location=(0, 0))

            renderer = p.circle(x=x.value, y=y.value, color={'field': color.value, 'transform': mapper},
                                size='sz',
                                line_color="lc",
                                line_width="lw",
                                line_alpha=0.9,
                                alpha=0.6, hover_color='white', hover_alpha=0.5, source=source)
            p.add_layout(color_bar, 'right')

        else:
            renderer = p.circle(x=x.value, y=y.value, size='sz',
                                line_color="lc",
                                line_width="lw",
                                line_alpha=0.9,
                                alpha=0.6,
                                hover_color='white', hover_alpha=0.5,
                                source=source)

        # for line in range(0, edges.shape[0]):
        #     p.line([edges.loc[line, 'from_x'], edges.loc[line, 'to_x']],
        #            [edges.loc[line, 'from_y'], edges.loc[line, 'to_y']],
        #            line_width=0.5, color='white')

        hover = HoverTool(
            tooltips=[
                ("index", "$index"),
                ("{}".format(size.value), "@{{{}}}".format(size.value)),
                ("{}".format(color.value), "@{{{}}}".format(color.value)),
                ("population", "@pop_names"),
                ("(x,y)", "($x, $y)"),
            ],
            renderers=[renderer]
        )
        p.add_tools(hover)
        draw_tool = PointDrawTool(renderers=[renderer], empty_value='black', add=False)
        p.add_tools(draw_tool)
        p.toolbar.active_tap = draw_tool

        new_columns = [
            TableColumn(field=x.value, title=x.value, formatter=formatter),
            TableColumn(field=y.value, title=y.value, formatter=formatter),
            TableColumn(field=color.value, title=color.value, formatter=formatter),
            TableColumn(field=size.value, title=size.value, formatter=formatter),
            TableColumn(field='pop_names', title="population"),
        ]
        data_table.columns = new_columns
        data_table.source = source
        layout.children[2] = DataTable(source=source, columns=new_columns, width=400, height=850)
        # data_table.update()

        return p

    p = figure(plot_height=900, plot_width=1200,
               tools='pan, box_zoom,reset, wheel_zoom, box_select, lasso_select,tap',
               toolbar_location="above")
    return p


# trying drawing using graphs, but missing easily moving of the points
def create_figure2(df):

    N = len(df)
    node_indices = list(range(1, N+1))

    plot = figure(title='Graph Layout Demonstration', x_range=(-1.1, 600), y_range=(-1.1, 600),
                  tools='pan, wheel_zoom, box_select', toolbar_location='above')

    graph = GraphRenderer()

    graph.node_renderer.data_source.add(node_indices, 'index')
    # graph.node_renderer.data_source.add(Spectral8, 'color')
    graph.node_renderer.glyph = Circle(radius=15)

    graph.selection_policy = NodesAndLinkedEdges()

    graph.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=1)
    graph.edge_renderer.data_source.data = dict(
        start=edges['edges.from'].tolist(),
        end=edges['edges.to'].tolist())

    # start of layout code
    x = df['x']
    y = df['y']
    graph_layout = dict(zip(node_indices, zip(x, y)))
    graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

    plot.renderers.append(graph)

    draw_tool = PointDrawTool(add=False, renderers=[graph])
    plot.add_tools(draw_tool)
    return plot


def update(attr, old, new):
    layout.children[1] = create_figure(df_patients)


def create_bubble():
    global populations
    # print(populations)
    populations = populations.append({'population_name': bubble_name.value,
                                      'color': population_colors.loc[len(populations), 'color_name']},
                                     ignore_index=True)
    selected = source.selected.indices
    df_patients.loc[selected, 'populationID'] = len(populations) - 1
    bubble_name.value = ""
    layout.children[1] = create_figure(df_patients)


def load_test_data():
    global patient_data
    global coordinates
    global edges
    global df_patients
    global source
    patient_data = pd.read_csv(join(dirname(__file__), 'data/pat_data.csv'))
    coordinates = pd.read_csv(join(dirname(__file__), 'data/coordinates.csv'))
    edges = pd.read_csv(join(dirname(__file__), 'data/edges.csv'))

    dropdown.button_type = "success"
    df_patients = hf.prepare_data(patient_data, coordinates)
    source = ColumnDataSource(df_patients)
    layout.children[1] = create_figure(df_patients)
    x.options = df_patients.columns.tolist()
    x.value = 'x'
    y.options = df_patients.columns.tolist()
    y.value = 'y'
    color.options = ['None'] + df_patients.columns.tolist()
    color.value = 'None'
    size.options = ['None'] + df_patients.columns.tolist()
    size.value = 'None'


file_source.on_change('data', file_callback)

test_data = Button(label="test data")
test_data.on_click(load_test_data)

x = Select(title='X-Axis', value='x', options=df_patients.columns.tolist())
y = Select(title='Y-Axis', value='y', options=df_patients.columns.tolist())
size = Select(title='Size', value='None', options=['None'] + df_patients.columns.tolist())
color = Select(title='Color', value='None', options=['None'] + df_patients.columns.tolist())

y.on_change('value', update)
x.on_change('value', update)
size.on_change('value', update)
color.on_change('value', update)

bubble_name = TextInput(placeholder="bubble's name", css_classes=['customTextInput'])
bubble = Button(label="Create bubble")
bubble.on_click(create_bubble)

menu = [("Upload patient data", "patient_data"), ("Upload cluster coordinates", "coordinates"),
        ("Upload graph edges", "edges")]
dropdown = Dropdown(label="Upload data", button_type="warning", menu=menu)
dropdown.callback = CustomJS(args=dict(file_source=file_source), code=up.file_read_callback)

controls = widgetbox([test_data, dropdown, x, y, color, size], width=200)

bubble_create = widgetbox([bubble_name, bubble], width=200)

formatter = NumberFormatter(format='0.0000')
data_table = DataTable(source=source, columns=[], width=400)

layout = row(column(controls, bubble_create), create_figure(df_patients), data_table)

curdoc().add_root(layout)
curdoc().title = "Flowexplore"
