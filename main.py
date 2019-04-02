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
from functools import reduce, partial
from math import pi
from io import StringIO, BytesIO
import base64

import help_functions as hf

# import sentry_sdk
# sentry_sdk.init("https://bc8203f867b04d5ca4e9129f144a192f@sentry.io/1406934")

file_source_tree = ColumnDataSource({'file_contents': [], 'file_name': []})

file_source_populations = ColumnDataSource({'file_contents': [], 'file_name': []})

file_source_patient = ColumnDataSource({'file_contents': [], 'file_name': []})

file_source_clinical = ColumnDataSource({'file_contents': [], 'file_name': []})

patients_data = {}

tree = {'coordinates': pd.DataFrame(), 'edges': pd.DataFrame()}

df_viz = pd.DataFrame()

populations = pd.DataFrame(columns=['population_name','color'])

source = ColumnDataSource()

population_colors = pd.read_csv(join(dirname(__file__), 'data/colors.csv'))  # TODO add more colors

clinical_data = pd.DataFrame()


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


def file_callback_populations(attr, old, new):  # TODO file check
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


def file_callback_clinical(attr, old, new):  # TODO file check
    global clinical_data

    filename = file_source_clinical.data['file_name'][0]
    raw_contents = file_source_clinical.data['file_contents'][0]

    # remove the prefix that JS adds
    prefix, b64_contents = raw_contents.split(",", 1)
    # print(b64_contents)
    file_contents = base64.b64decode(b64_contents)
    file_io = BytesIO(file_contents)
    # file_io = StringIO.StringIO(file_contents)
    # df = pd.read_excel(file_io)
    # file_io = StringIO(bytes.decode(file_contents))
    # print(file_io)

    clinical_data = pd.read_excel(file_io, header=[0, 1, 2])
    print(clinical_data)


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
        # callback = CustomJS(code="console.log('tap event occurred')")
        # p.js_on_event('lodend', callback)
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
    population = source.data["pop_names"][source.selected.indices[0]]
    new_indices = [i for i, g in enumerate(source.data["pop_names"]) if g == population]
    source.selected.indices = new_indices


def add_to_bubble(attr, old, new):
    indices = source.selected.indices
    if pop_list.value != 'None':
        df_viz.loc[indices, 'populationID'] = int(pop_list.value)
    else:
        df_viz.loc[indices, 'populationID'] = -1

    layout.children[1] = create_figure(df_viz, tree['edges'], populations)
    pop_list.value = 'placeholder'  # TODO find final solution


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
            x.value = 'x' if 'x' in df_viz.columns.tolist() else \
                (df_viz.columns.tolist()[0] if 'ID' not in df_viz.columns.tolist()[0] else df_viz.columns.tolist()[1])
            y.options = df_viz.columns.tolist()
            y.value = 'y' if 'y' in df_viz.columns.tolist() else \
                (df_viz.columns.tolist()[1] if 'ID' not in df_viz.columns.tolist()[1] else df_viz.columns.tolist()[2])
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


def check_selection(attr, old, new):
    IDs = set(list(source.data['populationID'][k] for k in source.selected.indices))
    if len(IDs) == 1:
        bubble_select.disabled = False
    else:
        bubble_select.disabled = True


def create_stats_tables():          # TODO remove error, if running without coordinates data
    global df_stats
    # create empty dataframe with multiindex
    bubbles = populations['population_name'].values
    markers = reduce(lambda s, q: s | q, map(lambda p: set(patients_data[p].columns.values), patients_data.keys()))
    iterables = [bubbles, markers]

    df_stats = pd.DataFrame(index=pd.MultiIndex.from_product(iterables), columns=patients_data.keys())
    # print(df_stats)
    marker.options = ['None'] + list(markers)

    for pat in patients_data:
        df_patient = patients_data[pat]

        # select column with cell count
        # TODO better condition to filter columns and find cell count
        int_cols = list(filter(lambda a: (not "id" in a.lower()) and df_patient[a].nunique() > 2,
                               df_patient.loc[:, df_patient.dtypes == np.int64].columns))
        # print(pat, int_cols)
        if len(int_cols) > 1:
            print("ERROR")      # TODO error message
        else:
            cell_count_column = int_cols[0]
            cell_sum = df_patient[cell_count_column].sum()
            # print(cell_sum)
            for idx_b, b in enumerate(bubbles):
                clusters = df_patient[df_viz['populationID'] == idx_b]
                for idx_m, m in enumerate(markers):
                    if m != cell_count_column and m in clusters.columns:
                        values = map(lambda a, count: a*clusters.loc[count, cell_count_column],
                                     clusters[m].dropna(), clusters[m].index.values)
                        df_stats.loc[(b, m), pat] = reduce(lambda p, q: p + q, list(values), 0) / cell_sum
                    else:
                        df_stats.loc[(b, m), pat] = reduce(lambda p, q: p + q, clusters[cell_count_column].values)
    df_stats = df_stats.astype(float)
    # print(df_stats)


def correlation_plot():
    if marker.value != "None":
        patients_list = list(patients_data.keys())

        p = figure(title="correlation on marker " + str(marker.value),
                   x_range=patients_list, y_range=patients_list,
                   plot_width=800, plot_height=800,
                   tools='pan, box_zoom,reset, wheel_zoom',
                   tooltips=[('rate', '@rate%')])

        p.grid.grid_line_color = None
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font_size = "10pt"
        p.axis.major_label_standoff = 0
        p.xaxis.major_label_orientation = pi / 2

        df = pd.DataFrame(index=patients_data[patients_list[0]].index.copy())

        for pat in patients_list:
            df[pat] = patients_data[pat][marker.value] if marker.value in patients_data[pat].columns else np.NaN

        df.columns = patients_list
        df = pd.DataFrame(df.corr().stack(), columns=['rate']).reset_index()
        # print(df)

        mapper = LinearColorMapper(palette=hf.create_color_map(),
                                   high=1,
                                   high_color='red',
                                   low=-1,
                                   low_color='blue'
                                   )
        color_bar = ColorBar(color_mapper=mapper, location=(0, 0))

        p.rect(x="level_0", y="level_1", width=1, height=1,
               source=df,
               fill_color={'field': 'rate', 'transform': mapper},
               line_color=None)
        p.add_layout(color_bar, 'right')
    else:
        p = figure(plot_height=800, plot_width=800,
                   tools='pan, box_zoom, reset, wheel_zoom',
                   toolbar_location="above")
    return p


def update_correlation_plot(attr, old, new):
    layout2.children[1] = correlation_plot()


def add_group():
    group_number = len(groups_tabs.tabs) + 1
    groups_tabs.tabs = groups_tabs.tabs + [create_panel(group_number)]


def remove_group():
    print(groups_tabs.active, " ", groups_tabs.tabs)
    groups_tabs.tabs.pop(groups_tabs.active)


def hold(attr, old, new):  # TODO add callback after pointDrawTool action
    print("eeee")


def rename_tab(text_input):
    global groups_tabs

    groups_tabs.tabs[groups_tabs.active].title = text_input.value
    text_input.value = ""
    new_tabs = Tabs(tabs=groups_tabs.tabs, active=groups_tabs.active)
    groups_tabs = new_tabs
    layout2.children[2].children[1].children[0] = new_tabs       # TODO time complexity???


def create_panel(group_number=1):       # TODO css classes
    remove_group_button = Button(label='remove group', width=200, button_type="danger")
    remove_group_button.on_click(remove_group)

    group_name = TextInput(placeholder="rename group")
    confirm = Button(label="OK", width=200)
    confirm.on_click(partial(rename_tab, text_input=group_name))

    edit_row = row([remove_group_button, group_name, confirm])
    new_tab = Panel(child=edit_row, title="group " + str(group_number), closable=True)
    return new_tab


# TAB1 population view ----------------------------------------------------------------------- TAB1 population view

# file loading and update
file_source_tree.on_change('data', file_callback_tree)

file_source_populations.on_change('data', file_callback_populations)

file_source_patient.on_change('data', file_callback_pat)

file_source_clinical.on_change('data', file_callback_clinical)

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
bubble_select = Button(label='select the whole population', button_type="primary", disabled=True)
bubble_select.on_click(select_population)

# if selection change, check bubble_select button
source.selected.on_change('indices', check_selection)

# add selected to a population
pop_list = Dropdown(label='add selected to a bubble', menu=[('None', 'None')])
pop_list.on_change('value', add_to_bubble)

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

create_bubble_stats = Button(label="create tables for statistics")
create_bubble_stats.on_click(create_stats_tables)
marker = Select(title='Marker', value='None', options=['None'])
marker.on_change('value', update_correlation_plot)

basic_overview = widgetbox([create_bubble_stats, marker], width=200)

add_group_button = Button(label='Add new group', width=200)
add_group_button.on_click(add_group)

upload_clinical_data = Button(label='upload clinical data', width=200)
upload_clinical_data.js_on_click(CustomJS(args=dict(file_source=file_source_clinical),
                                          code=open(join(dirname(__file__), "static/js/upload.js")).read()))

groups_tabs = Tabs(tabs=[create_panel()])
groups_tabs.width = 800

layout2 = row(basic_overview, correlation_plot(), column(children=[row(add_group_button, upload_clinical_data),
                                                                   groups_tabs]))

tab2 = Panel(child=layout2, title="group selection view")

# TAB3 test results ------------------------------------------------------------------------ TAB3 test results

c = Button(label="under development")

tab3 = Panel(child=c, title="statistics view")

# FINAL LAYOUT ------------------------------------------------------------------------------------- FINAL LAYOUT

tabs = Tabs(tabs=[tab1, tab2, tab3])
curdoc().add_root(tabs)
curdoc().title = "Flowexplore"
