import pandas as pd
import numpy as np
from os.path import join, dirname
from bokeh.layouts import row, widgetbox, column
from bokeh.models import Select, ColorBar, ColumnDataSource, HoverTool, PointDrawTool, \
    CustomJS, LassoSelectTool, GraphRenderer, StaticLayoutProvider, Circle, MultiLine
from bokeh.models.widgets import Button, Dropdown, TextInput, DataTable, TableColumn, NumberFormatter, Panel, Tabs
from bokeh.models.mappers import LinearColorMapper
from bokeh.models.graphs import NodesAndLinkedEdges
from bokeh.models.selections import Selection
from bokeh.plotting import curdoc, figure
from functools import reduce

from io import StringIO
import base64

import help_functions as hf

# import pyreadr

# import sentry_sdk
# sentry_sdk.init("https://bc8203f867b04d5ca4e9129f144a192f@sentry.io/1406934")

file_source_tree = ColumnDataSource({'file_contents': [], 'file_name': []})

file_source_populations = ColumnDataSource({'file_contents': [], 'file_name': []})

file_source_patient = ColumnDataSource({'file_contents': [], 'file_name': []})

patients_data = {}

tree = {'coordinates': pd.DataFrame(), 'edges': pd.DataFrame()}

df_viz = pd.DataFrame()

populations = pd.DataFrame()

source = ColumnDataSource()

population_colors = pd.read_csv(join(dirname(__file__), 'data/colors.csv'))  # TODO add more colors


def file_callback_tree(attr, old, new):  # TODO file check
    global df_viz
    global source

    filename = file_source_tree.data['file_name'][0]
    raw_contents = file_source_tree.data['file_contents'][0]

    # remove the prefix that JS adds
    prefix, b64_contents = raw_contents.split(",", 1)
    file_contents = base64.b64decode(b64_contents)
    file_io = StringIO(bytes.decode(file_contents))
    df = pd.read_csv(file_io)
    # print("file contents:")
    # print(df)
    if tree_dropdown.value == 'coordinates':
        tree['coordinates'] = df
        tree_dropdown.menu[0] = ("coordinates ok (" + filename + ")", 'coordinates')
        df_viz['x'] = tree['coordinates'].iloc[:, 1].values
        df_viz['y'] = tree['coordinates'].iloc[:, 2].values
        df_viz['populationID'] = -1
        source.data = df_viz.to_dict(orient='list')
        layout.children[1] = create_figure(df_viz, tree['edges'], populations)
        x.options = df_viz.columns.tolist()
        x.value = 'x'
        y.options = df_viz.columns.tolist()
        y.value = 'y'
        color.options = ['None'] + df_viz.columns.tolist()
        color.value = 'None'
        size.options = ['None'] + df_viz.columns.tolist()
        size.value = 'None'

    elif tree_dropdown.value == 'edges':
        tree['edges'] = df
        tree_dropdown.menu[1] = ("edges ok (" + filename + ")", 'edges')
        layout.children[1] = create_figure(df_viz, tree['edges'], populations)
    else:
        print("something went wrong, unknown dropdown value")  # TODO error message?
    if reduce(lambda a, q: a and q, [True if 'ok' in string[0] else False for string in tree_dropdown.menu]):
        tree_dropdown.button_type = "success"


def file_callback_populations(attr, old, new):      # TODO file check
    global df_viz
    global populations

    raw_contents = file_source_populations.data['file_contents'][0]

    # remove the prefix that JS adds
    prefix, b64_contents = raw_contents.split(",", 1)
    file_contents = base64.b64decode(b64_contents)
    file_io = StringIO(bytes.decode(file_contents))

    text = list(iter(file_io.getvalue().splitlines()))
    df_viz['populationID'] = -1
    populations = pd.DataFrame()
    for line in text:
        if line != "":
            split_line = line.split(":")
            pop_name = split_line[0]

            populations = populations.append({'population_name': pop_name,
                                              'color': population_colors.loc[len(populations), 'color_name']},
                                             ignore_index=True)
            pop_list.menu.append((pop_name, str(len(populations) - 1)))

            indices = [int(a) for a in split_line[1].split(",")]
            # print(indices)

            df_viz.loc[indices, 'populationID'] = len(populations) - 1
            patches = {
                'populationID': [(i, len(populations) - 1) for i in indices]
            }
            source.patch(patches)
            bubble_name.value = ""

    upload_populations.button_type = 'success'
    layout.children[1] = create_figure(df_viz, tree['edges'], populations)


def file_callback_pat(attr, old, new):  # TODO file check, upload population data
    global df_viz
    global source
    global populations

    filename = file_source_patient.data['file_name'][-1]
    raw_contents = file_source_patient.data['file_contents'][-1]

    # remove the prefix that JS adds
    prefix, b64_contents = raw_contents.split(",", 1)
    file_contents = base64.b64decode(b64_contents)
    file_io = StringIO(bytes.decode(file_contents))
    # print("file contents:")
    # print(df)
    df = pd.read_csv(file_io)
    # print(filename.split(".")[0])
    ind = filename.split(".")[0]
    if 'Unnamed: 0' in df.columns:  # TODO drop all Unnamed
        df.drop(columns=['Unnamed: 0'], inplace=True)
    patients_data[ind] = df
    patient.options = patient.options + [ind]
    patient.value = ind
    upload_patients.button_type = 'success'


def create_figure(df, df_edges, df_populations):
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

        # add lines
        # print(source.data['x'][0])
        if not df_edges.empty:
            lines_from = []
            lines_to = []
            for line in range(0, df_edges.shape[0]):
                lines_from.append(
                    [source.data[x.value][df_edges.iloc[line, 1] - 1],  # TODO filter possible nan values
                     source.data[x.value][df_edges.iloc[line, 2] - 1]])
                lines_to.append([source.data[y.value][df_edges.iloc[line, 1] - 1],  # TODO filter possible nan values
                                 source.data[y.value][df_edges.iloc[line, 2] - 1]])

            p.multi_line(lines_from, lines_to, line_width=0.5, color='white')
            # lines_renderer = p.multi_line(lines_from, lines_to, line_width=0.5, color='white')
        # else:
            # lines_renderer = None

        # mark populations
        line_color = ['white'] * len(df)
        line_width = [1] * len(df)
        if not df_populations.empty:
            line_color = [populations.iloc[pop_id]['color'] if pop_id != -1 else 'white'
                          for pop_id in df['populationID']]
            line_width = [5 if lc != 'white' else 1 for lc in line_color]

        source.add(line_color, name='lc')
        source.add(line_width, name='lw')

        if size.value != 'None':
            sizes = [hf.scale(value, df[size.value].min(),
                              df[size.value].max()) if not np.isnan(value) and value != 0 else 10 for value in
                     df[size.value]]
        else:
            sizes = [25 for _ in df[x.value]]
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
        draw_tool = PointDrawTool(renderers=[renderer], add=False)
        # draw_tool = PointDrawTool(renderers=[renderer], add=False)
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

        # download new coordinates
        if 'x' in df_viz.columns.tolist():
            download.callback = CustomJS(args=dict(source=source,
                                                   columns=" ".join(['x', 'y', 'pop_names']),
                                                   num_of_columns=3),
                                         code=open(join(dirname(__file__), "static/js/download.js")).read())

        # download population list with cluster numbers
        if 'population_name' in populations.columns:
            text = ""
            for index, population in populations.iterrows():
                text += population['population_name'] + ": " + \
                        ", ".join(str(e) for e in df_viz.index[df_viz['populationID'] == index].tolist()) + "\n\n"
                # print(text)
                download_populations.callback = CustomJS(args=dict(text=text),
                                                         code=open(join(dirname(__file__),
                                                                        "static/js/download_populations.js")).read())

        return p

    p = figure(plot_height=900, plot_width=1200,
               tools='pan, box_zoom,reset, wheel_zoom, box_select, lasso_select,tap',
               toolbar_location="above")
    return p


# trying drawing using graphs, but missing easily moving of vertices
def create_figure2(df):
    N = len(df)
    node_indices = list(range(1, N + 1))

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
    xx = df['x']
    yy = df['y']
    graph_layout = dict(zip(node_indices, zip(xx, yy)))
    graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

    plot.renderers.append(graph)

    draw_tool = PointDrawTool(add=False, renderers=[graph])
    plot.add_tools(draw_tool)
    return plot


def update(attr, old, new):
    layout.children[1] = create_figure(df_viz, tree['edges'], populations)


def create_bubble():
    global populations
    global source
    # print(populations)
    populations = populations.append({'population_name': bubble_name.value,
                                      'color': population_colors.loc[len(populations), 'color_name']},
                                     ignore_index=True)
    pop_list.menu.append((bubble_name.value, str(len(populations) - 1)))
    selected = source.selected.indices
    df_viz.loc[selected, 'populationID'] = len(populations) - 1
    patches = {
        'populationID': [(i, len(populations) - 1) for i in selected]
    }
    source.patch(patches)
    bubble_name.value = ""
    layout.children[1] = create_figure(df_viz, tree['edges'], populations)


def load_test_data():
    global patient_data
    global coordinates
    global edges
    global df_viz
    global source

    patient_data = pd.read_csv(join(dirname(__file__), 'data/patient_36.csv'))
    patient_data2 = pd.read_csv(join(dirname(__file__), 'data/patient_38.csv'))
    coordinates = pd.read_csv(join(dirname(__file__), 'data/coordinates.csv'))
    edges = pd.read_csv(join(dirname(__file__), 'data/edges.csv'))

    tree['coordinates'] = coordinates
    tree_dropdown.menu[0] = ("coordinates ok (coordinates.csv)", 'coordinates')
    df_viz['x'] = tree['coordinates'].iloc[:, 1].values
    df_viz['y'] = tree['coordinates'].iloc[:, 2].values
    df_viz['populationID'] = -1
    source.data = df_viz.to_dict(orient='list')

    tree['edges'] = edges
    tree_dropdown.menu[1] = ("edges ok (edges.csv)", 'edges')
    tree_dropdown.button_type = 'success'

    ind = "36"
    if 'Unnamed: 0' in patient_data.columns:
        patient_data.drop(columns=['Unnamed: 0'], inplace=True)
    patients_data[ind] = patient_data
    patient.options = patient.options + [ind]
    # patient.value = ind/

    ind = "38"
    if 'Unnamed: 0' in patient_data2.columns:
        patient_data2.drop(columns=['Unnamed: 0'], inplace=True)
    patients_data[ind] = patient_data2
    patient.options = patient.options + [ind]
    # patient.value = ind

    # layout.children[1] = create_figure(df_viz, tree['edges'], populations)
    x.options = df_viz.columns.tolist()
    x.value = 'x'
    y.options = df_viz.columns.tolist()
    y.value = 'y'
    color.options = ['None'] + df_viz.columns.tolist()
    color.value = 'None'
    size.options = ['None'] + df_viz.columns.tolist()
    size.value = 'None'

    layout.children[1] = create_figure(df_viz, tree['edges'], populations)


def select_population():
    indices = source.selected.indices
    if len(indices) == 1:
        population = source.data["pop_names"][indices[0]]
        new_indices = [i for i, g in enumerate(source.data["pop_names"]) if g == population]
        if new_indices != indices:
            source.selected = Selection(indices=new_indices)
    else:
        print("WARNING: SELECT ONLY ONE NODE")  # TODO create warning message in UI!


def add_to_bubble(attr, old, new):
    indices = source.selected.indices
    if pop_list.value != 'None':
        df_viz.loc[indices, 'populationID'] = int(pop_list.value)
    else:
        df_viz.loc[indices, 'populationID'] = -1

    layout.children[1] = create_figure(df_viz, tree['edges'], populations)
    pop_list.value = 'placeholder'      # TODO find final solution


def select_patient(attr, old, new):
    global df_viz
    global source
    if patient.value != 'None':
        if df_viz.empty:
            df_viz = patients_data[patient.value]
            df_viz['populationID'] = -1
            source.data = df_viz.to_dict(orient='list')
            x.options = df_viz.columns.tolist()
            x.value = df_viz.columns.tolist()[0]
            y.options = df_viz.columns.tolist()
            y.value = df_viz.columns.tolist()[1]
            color.options = ['None'] + df_viz.columns.tolist()
            color.value = 'None'
            size.options = ['None'] + df_viz.columns.tolist()
            size.value = 'None'
        else:
            drop_cols = map(lambda a: a if a not in ['x', 'y', 'populationID'] else None, df_viz.columns.tolist())
            df_viz.drop(columns=filter(lambda v: v is not None, drop_cols), inplace=True)
            df_viz = df_viz.join(patients_data[patient.value])
            # print(df_viz)
            source.data = df_viz.to_dict(orient='list')
            x.options = df_viz.columns.tolist()
            x.value = 'x' if 'x' in df_viz.columns.tolist() else df_viz.columns.tolist()[0]
            y.options = df_viz.columns.tolist()
            y.value = 'y' if 'y' in df_viz.columns.tolist() else df_viz.columns.tolist()[1]
            color.options = ['None'] + df_viz.columns.tolist()
            color.value = 'None'
            size.options = ['None'] + df_viz.columns.tolist()
            size.value = 'None'
    else:
        if 'x' not in df_viz.columns.tolist():
            df_viz = pd.DataFrame()
            source.data = {}
        else:
            drop_cols = map(lambda a: a if a not in ['x', 'y', 'populationID'] else None, df_viz.columns.tolist())
            df_viz.drop(columns=filter(lambda v: v is not None, drop_cols), inplace=True)
    layout.children[1] = create_figure(df_viz, tree['edges'], populations)


def create_stats_tables():
    for pat in patients_data:
        count = patients_data[pat].iloc[:, 1].values

        for pop in range(len(populations)):
            # clusters = patients_data[pat][patients_data[pat]['populationID'] == pop]
            print(patients_data[pat])
            print()


# TAB1 population view ----------------------------------------------------------------------- TAB1 population view

# file loading and update
file_source_tree.on_change('data', file_callback_tree)

file_source_populations.on_change('data', file_callback_populations)

file_source_patient.on_change('data', file_callback_pat)

# test data loading, only for testing
# test_data = Button(label="test data")
# test_data.on_click(load_test_data)

# upload tree files
menu_tree = [("Upload cluster coordinates", "coordinates"), ("Upload graph edges", "edges")]
tree_dropdown = Dropdown(label="Upload tree structure", button_type="warning", menu=menu_tree)
tree_dropdown.callback = CustomJS(args=dict(file_source=file_source_tree),
                                  code=open(join(dirname(__file__), "static/js/upload.js")).read())

# upload patients data
upload_patients = Button(label="upload patients data")
upload_patients.js_on_click(CustomJS(args=dict(file_source=file_source_patient),
                                     code=open(join(dirname(__file__), "static/js/upload_multiple.js")).read()))

# upload population list
upload_populations = Button(label="upload population list")
upload_populations.js_on_click(CustomJS(args=dict(file_source=file_source_populations),
                                        code=open(join(dirname(__file__), "static/js/upload.js")).read()))

# select patient
patient = Select(title='Patient', value='None', options=['None'] + df_viz.columns.tolist())

patient.on_change('value', select_patient)

# interaction with the plot
x = Select(title='X-Axis', value='x', options=df_viz.columns.tolist())
y = Select(title='Y-Axis', value='y', options=df_viz.columns.tolist())
size = Select(title='Size', value='None', options=['None'] + df_viz.columns.tolist())
color = Select(title='Color', value='None', options=['None'] + df_viz.columns.tolist())

y.on_change('value', update)
x.on_change('value', update)
size.on_change('value', update)
color.on_change('value', update)

# create bubbles
bubble_name = TextInput(placeholder="bubble's name", css_classes=['customTextInput'])
bubble = Button(label="Create bubble")
bubble.on_click(create_bubble)

# select population
bubble_select = Button(label='select the whole population', button_type="primary")
bubble_select.on_click(select_population)

# add selected to a population
pop_list = Dropdown(label='add selected to a bubble',  menu=[('None', 'None')])
pop_list.on_change('value', add_to_bubble)

# TODO selected on change callback
# source.selected.js_on_change('indices', CustomJS(args=dict(source=source, button=bubble_select), code="""
#                             console.log('aaaa');
#                             // button.disabled = true;
#                             """)
#                              )

# download data new coordinates
download = Button(label="download tree structure", button_type="primary")

# download the populations
download_populations = Button(label="download population list", button_type="primary")

# controls = widgetbox([test_data, tree_dropdown, upload_patients, upload_populations,
controls = widgetbox([tree_dropdown, upload_patients, upload_populations,
                      patient, x, y, color, size], width=200)

add_bubble_box = widgetbox([bubble_name, bubble], width=200, css_classes=['bubbles'])
bubble_tools_box = widgetbox([bubble_select, pop_list], width=200, css_classes=['bubbles2'])

download_tools_box = widgetbox([download, download_populations], width=200)

# data table
formatter = NumberFormatter(format='0.0000')
data_table = DataTable(source=source, columns=[], width=400, height=850, reorderable=True)

layout = row(column(controls, add_bubble_box, bubble_tools_box, download_tools_box),
             create_figure(df_viz, tree['edges'], populations),
             data_table)

tab1 = Panel(child=layout, title="population view")

# TAB2 group selection ----------------------------------------------------------------------- TAB2 group selection

create_bubble_stats = Button(label="under development", width=200)
create_bubble_stats.on_click(create_stats_tables)

tab2 = Panel(child=create_bubble_stats, title="group selection view")

# TAB3 test results ------------------------------------------------------------------------ TAB3 test results

c = Button(label="under development")

tab3 = Panel(child=c, title="statistics view")

# FINAL LAYOUT ------------------------------------------------------------------------------------- FINAL LAYOUT

tabs = Tabs(tabs=[tab1, tab2, tab3])
curdoc().add_root(tabs)
curdoc().title = "Flowexplore"
