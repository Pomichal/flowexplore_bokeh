import pandas as pd
import numpy as np
from os.path import join, dirname
from bokeh.layouts import row, widgetbox, column
from bokeh.models import Select, ColorBar, ColumnDataSource, HoverTool, PointDrawTool, \
    CustomJS, LassoSelectTool, GraphRenderer, StaticLayoutProvider, Circle, MultiLine
from bokeh.models.widgets import Button, Dropdown, TextInput, DataTable, TableColumn, NumberFormatter, Panel, Tabs, \
    PreText, DateRangeSlider, RangeSlider, CheckboxGroup, Div, MultiSelect
from bokeh.models.mappers import LinearColorMapper
from bokeh.plotting import curdoc, figure
from functools import reduce, partial
from math import pi
from io import StringIO, BytesIO
import base64

from help_functions import help_functions as hf
from help_functions import boxplot
from help_functions import file_upload
from help_functions import create_figure, manipulate_figure, correlation_plot, manipulate_groups, diff_plot

file_source_tree = ColumnDataSource({'file_contents': [], 'file_name': []})

file_source_populations = ColumnDataSource({'file_contents': [], 'file_name': []})

file_source_patient = ColumnDataSource({'file_contents': [], 'file_name': []})

file_source_clinical = ColumnDataSource({'file_contents': [], 'file_name': []})

patients_data = {}

tree = {'coordinates': pd.DataFrame(), 'edges': pd.DataFrame()}

df_viz = pd.DataFrame()

populations = pd.DataFrame(columns=['population_name', 'color'])

source = ColumnDataSource()

population_colors = pd.read_csv(join(dirname(__file__), 'data/colors.csv'))  # TODO add more colors

clinical_data = pd.DataFrame()

groups = []  # conditions for groups of patients


def file_callback_tree(attr, old, new):  # TODO file check
    global df_viz
    global source
    global tree

    df_viz, tree = file_upload.file_callback_tree(file_source_tree, tree_dropdown.value, df_viz, tree, source)

    if tree_dropdown.value == 'coordinates':
        tree_dropdown.menu[0] = ("coordinates ok (" + file_source_tree.data['file_name'][0] + ")", 'coordinates')
        x.options = df_viz.columns.tolist()
        x.value = 'x'
        y.options = df_viz.columns.tolist()
        y.value = 'y'
        color.options = ['None'] + df_viz.columns.tolist()
        color.value = 'None'
        size.options = ['None'] + df_viz.columns.tolist()
        size.value = 'None'

    elif tree_dropdown.value == 'edges':
        tree_dropdown.menu[1] = ("edges ok (" + file_source_tree.data['file_name'][0] + ")", 'edges')
    else:
        print("something went wrong, unknown dropdown value")  # TODO error message?

    layout.children[1] = draw_figure(df_viz, tree['edges'], populations)
    if reduce(lambda a, q: a and q, [True if 'ok' in string[0] else False for string in tree_dropdown.menu]):
        tree_dropdown.button_type = "success"


def file_callback_populations(attr, old, new):  # TODO file check
    global df_viz
    global populations
    global source

    df_viz, populations, source = file_upload.file_callback_populations(file_source_populations, df_viz, source)

    pop_list.menu = [('None', 'None')] + [(name, str(index))
                                          for index, name in enumerate(populations['population_name'])]
    upload_populations.button_type = 'success'
    layout.children[1] = draw_figure(df_viz, tree['edges'], populations)


def file_callback_pat(attr, old, new):  # TODO file check, upload population data

    df, name = file_upload.file_callback_pat(file_source_patient)
    patients_data[name] = df
    patient.options = patient.options + [name]
    patient.value = name
    upload_patients.button_type = 'success'


def file_callback_clinical(attr, old, new):  # TODO file check
    global clinical_data

    clinical_data = file_upload.file_callback_clinical(file_source_clinical)

    upload_clinical_data.button_type = 'success'
    groups_tabs.tabs[0] = create_panel()

    # groups[0][1] = manipulate_groups.map_measurements_to_patients(c_data=clinical_data,
    #                                                               pats_data=patients_data),
    add_group_button.disabled = False
    create_ref_group_button.disabled = False


def draw_figure(df, df_edges, df_populations):

    fig, new_columns = create_figure.create_figure(df, df_edges, df_populations, source, x.value, y.value, color.value, size.value)
    if not df.empty:

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

    return fig


def update(attr, old, new):
    layout.children[1] = draw_figure(df_viz, tree['edges'], populations)


def create_bubble():
    global populations
    global source
    global df_viz

    df_viz, populations = manipulate_figure.create_bubble(source, populations, bubble_name.value, df_viz)

    pop_list.menu.append((bubble_name.value, str(len(populations) - 1)))

    bubble_name.value = ""
    layout.children[1] = draw_figure(df_viz, tree['edges'], populations)


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

    layout.children[1] = draw_figure(df_viz, tree['edges'], populations)
    pop_list.value = 'placeholder'  # TODO find final solution


def select_patient(attr, old, new):
    global df_viz
    global source

    if patient.value != 'None':
        df_viz = manipulate_figure.select_patient(df_viz, source, patient.value, patients_data[patient.value])
    else:
        df_viz = manipulate_figure.select_patient(df_viz, source, patient.value, pd.DataFrame())

    if not df_viz.empty:
        x.options = df_viz.columns.tolist()
        x.value = df_viz.columns.tolist()[0]
        y.options = df_viz.columns.tolist()
        y.value = df_viz.columns.tolist()[1]
        color.options = ['None'] + df_viz.columns.tolist()
        color.value = 'None'
        size.options = ['None'] + df_viz.columns.tolist()
        size.value = 'None'
        x.value = 'x' if 'x' in df_viz.columns.tolist() else \
            (df_viz.columns.tolist()[0] if 'ID' not in df_viz.columns.tolist()[0] else df_viz.columns.tolist()[1])
        y.options = df_viz.columns.tolist()
        y.value = 'y' if 'y' in df_viz.columns.tolist() else \
            (df_viz.columns.tolist()[-1] if 'ID' not in df_viz.columns.tolist()[-1] else df_viz.columns.tolist()[
                -2])
    else:   # TODO show error message?
        print("ERROR: please select a patient")
    layout.children[1] = draw_figure(df_viz, tree['edges'], populations)


def check_selection(attr, old, new):
    IDs = set(list(source.data['populationID'][k] for k in source.selected.indices))
    if len(IDs) == 1:
        bubble_select.disabled = False
    else:
        bubble_select.disabled = True


def create_stats_tables():  # TODO remove error, if running without coordinates data
    global df_stats

    df_stats, marker_list = hf.create_stats_tables(populations, patients_data, df_viz)

    if df_stats.empty:
        print("ERROR, couldn't find the cell count column")  # TODO error message?
    else:
        marker.options = ['None'] + list(marker_list)
        marker.value = marker.options[-1]


def update_correlation_plot(attr, old, new):
    layout2.children[1] = correlation_plot.correlation_plot(marker.value, patients_data)


def create_reference_group_tab():
    group_number = len(groups_tabs.tabs)

    groups.append([{}, manipulate_groups.map_measurements_to_patients(c_data=clinical_data, pats_data=patients_data)])
    groups[group_number][1] = hf.find_healthy(patients_data, clinical_data)
    new_tab = manipulate_groups.create_reference_group_tab(group_number,
                                                           groups, remove_group, rename_tab, remove_measurements)

    groups_tabs.tabs = groups_tabs.tabs + [new_tab]
    groups_tabs.active = len(groups_tabs.tabs) - 1


def remove_measurements_reference_group():
    group_no = groups_tabs.active
    indices = groups_tabs.tabs[groups_tabs.active].child.children[0].source.selected.indices
    groups[groups_tabs.active][1].drop(groups[groups_tabs.active][1].index[indices], inplace=True)
    groups_tabs.tabs[group_no].child.children[0].source = ColumnDataSource(groups[group_no][1])


def remove_measurements():
    group_no = groups_tabs.active
    indices = groups_tabs.tabs[groups_tabs.active].child.children[2].children[0].source.selected.indices
    groups[groups_tabs.active][1].drop(groups[groups_tabs.active][1].index[indices], inplace=True)
    groups_tabs.tabs[group_no].child.children[2].children[0].source = ColumnDataSource(groups[group_no][1])


def add_group():
    group_number = len(groups_tabs.tabs)
    groups.append([{}, manipulate_groups.map_measurements_to_patients(c_data=clinical_data, pats_data=patients_data)])
    groups_tabs.tabs = groups_tabs.tabs + [create_panel(group_number)]
    groups_tabs.active = len(groups_tabs.tabs) - 1


def remove_group():
    # global merge
    groups_tabs.tabs.pop(groups_tabs.active)
    groups.pop(groups_tabs.active)
    # print(merge.options.pop(groups_tabs.active + 1))
    # li = merge.options                                        # TODO merge
    # del li[groups_tabs.active + 1]
    # merge.options = li


def hold(attr, old, new):  # TODO add callback after pointDrawTool action
    print("eeee")


def rename_tab(text_input):
    global groups_tabs

    groups_tabs.tabs[groups_tabs.active].title = text_input.value
    text_input.value = ""
    new_tabs = Tabs(tabs=groups_tabs.tabs, active=groups_tabs.active)
    groups_tabs = new_tabs
    layout2.children[2].children[1] = new_tabs  # TODO time complexity???


def create_panel(group_number=0):  # TODO css classes

    return manipulate_groups.create_panel(clinical_data, patients_data, rename_tab, remove_group,
                                          remove_measurements, groups, group_number)


def active_tab(attr, old, new):
    if old == 0 and new == 1:
        create_stats_tables()
    if new == 2:
        bubbles.options = ["None"] + df_stats.index.get_level_values(0).unique().tolist()
        markers.options = ["None"] + df_stats.index.get_level_values(1).unique().tolist()
        # print(df_stats)
        bubbles.value = bubbles.options[1]
        markers.value = markers.options[1]


def draw_boxplot(attr, old, new):
    b = bubbles.value
    m = markers.value
    # print(b, m)
    if b != 'None' and m != 'None':
        calculate_diff(b,m)
        # print(groups)
        stats_df = pd.DataFrame(df_stats.loc[(b, m), :])
        # print(stats_df)
        for i in stats_df.index:
            # print(i)
            for group_number, group in enumerate(groups):
                # print(group)
                # print(group[1])
                # print(group[1]['measurements'])
                if i in group[1]['measurements'].tolist():
                    # print(i, group_number, group)
                    stats_df.loc[i, 'group'] = groups_tabs.tabs[group_number].title
        # print(stats_df)
        layout3.children[1] = boxplot.create_boxplot(stats_df)


def calculate_diff(b, m):
    stats_df = pd.DataFrame(df_stats.loc[(b, m), :])
    diff_df = pd.DataFrame(index=list([tab.title for tab in groups_tabs.tabs]), columns=['diff'])

    reference_level = 0
    for g in groups:
        measurements = g[1]['measurements'].tolist()
        if g[1]['patient'][0] == 'healthy':
            reference_level = np.mean(list([stats_df.loc[measurement, (b, m)] for measurement in measurements]))
            break

    for idx, g in enumerate(groups):
        measurements = g[1]['measurements'].tolist()
        group_level = np.mean(list([stats_df.loc[measurement, (b, m)] for measurement in measurements]))
        diff_df.loc[groups_tabs.tabs[idx].title, 'diff'] = group_level - reference_level

    layout3.children[2] = diff_plot.diff_plot(diff_df)


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
tree_dropdown = Dropdown(label="Upload tree structure", button_type="warning", menu=menu_tree,
                         css_classes=['dropdowns'])
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
pop_list = Dropdown(label='add selected to a bubble', menu=[('None', 'None')],
                    css_classes=['dropdowns'])
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
data_table = DataTable(source=source, columns=[], width=400, height=850, reorderable=True)

layout = row(column(controls, add_bubble_box, bubble_tools_box, download_tools_box),
             draw_figure(df_viz, tree['edges'], populations),
             data_table)

tab1 = Panel(child=layout, title="population view")

# TAB2 group selection ----------------------------------------------------------------------- TAB2 group selection

# create_bubble_stats = Button(label="create tables for statistics")
# create_bubble_stats.on_click(create_stats_tables)
marker = Select(title='Marker', value='None', options=['None'])
marker.on_change('value', update_correlation_plot)

basic_overview = widgetbox([
    # create_bubble_stats,
    marker], width=200)

add_group_button = Button(label='Add new group', width=200, disabled=True)
add_group_button.on_click(add_group)

create_ref_group_button = Button(label="Create reference group", width=200, disabled=True)
create_ref_group_button.on_click(create_reference_group_tab)

upload_clinical_data = Button(label='upload clinical data', width=200)
upload_clinical_data.js_on_click(CustomJS(args=dict(file_source=file_source_clinical),
                                          code=open(join(dirname(__file__), "static/js/upload.js")).read()))

# merge = Select(title='Merge current group with:', value='None', options=['None'], width=200)      # TODO merge

group1 = create_panel()
groups.append([{}, manipulate_groups.map_measurements_to_patients(c_data=clinical_data, pats_data=patients_data)])

groups_tabs = Tabs(tabs=[group1])
groups_tabs.width = 800

layout2 = row(basic_overview, correlation_plot.correlation_plot(marker.value, patients_data),
              column(children=[column(row(add_group_button, create_ref_group_button, upload_clinical_data),
                                                                          # merge
                                      ),
                               groups_tabs]))

tab2 = Panel(child=layout2, title="group selection view")

# TAB3 test results ------------------------------------------------------------------------ TAB3 test results

c = Button(label="under development")

bubbles = Select(title='Population', value='None', options=['None'])
markers = Select(title='Marker', value='None', options=['None'])

bubbles.on_change('value', draw_boxplot)
markers.on_change('value', draw_boxplot)

layout3 = row(column(bubbles, markers), boxplot.create_boxplot(), diff_plot.diff_plot())

tab3 = Panel(child=layout3, title="statistics view")

# FINAL LAYOUT ------------------------------------------------------------------------------------- FINAL LAYOUT

tabs = Tabs(tabs=[tab1, tab2, tab3])
tabs.on_change('active', active_tab)
curdoc().add_root(tabs)
curdoc().title = "Flowexplore"


def load_test_data():
    global df_viz
    global source
    global populations
    global clinical_data

#################################################################### coordinates

    filename = 'coordinates.csv'
    df = pd.read_csv(join(dirname(__file__), 'data/coordinates.csv'))
    # if tree_dropdown.value == 'coordinates':
    tree['coordinates'] = df
    tree_dropdown.menu[0] = ("coordinates ok (" + filename + ")", 'coordinates')
    df_viz['x'] = tree['coordinates'].iloc[:, 1].values
    df_viz['y'] = tree['coordinates'].iloc[:, 2].values
    df_viz['populationID'] = -1
    source.data = df_viz.to_dict(orient='list')
    layout.children[1] = draw_figure(df_viz, tree['edges'], populations)
    x.options = df_viz.columns.tolist()
    x.value = 'x'
    y.options = df_viz.columns.tolist()
    y.value = 'y'
    color.options = ['None'] + df_viz.columns.tolist()
    color.value = 'None'
    size.options = ['None'] + df_viz.columns.tolist()
    size.value = 'None'

##################################################################### edges
    filename = 'edges.csv'
    df = pd.read_csv(join(dirname(__file__), 'data/edges.csv'))

    tree['edges'] = df
    tree_dropdown.menu[1] = ("edges ok (" + filename + ")", 'edges')
    layout.children[1] = draw_figure(df_viz, tree['edges'], populations)

    # if reduce(lambda a, q: a and q, [True if 'ok' in string[0] else False for string in tree_dropdown.menu]):
    tree_dropdown.button_type = "success"

##################################################################### patients
    pat_list = [
        'pBM-36-sort-CD15-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 62 .csv',
        'pBM-36-sort-CD15-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 63 .csv',
        'pBM-38-unsort-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 191 .csv',
        'pBM-38-unsort-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 192 .csv',
        # 'pBM-39-unsort-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 193 .csv',
        # 'pBM-39-unsort-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 194 .csv',
        # 'pBM-41-sort-CD15-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 195 .csv',
        # 'pBM-41-sort-CD15-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 196 .csv',
        # 'pBM-43-sort-CD15-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 197 .csv',
        # 'pBM-43-sort-CD15-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 198 .csv',
        # 'pBM-48-sort-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 199 .csv',
        # 'pBM-48-sort-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 200 .csv',
        # 'pBM-49-sort-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 201 .csv',
        # 'pBM-49-sort-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 202 .csv',
        'hBM-1n-sort-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 1 .csv',
        'hBM-1n-sort-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 2 .csv',
        'hBM-1n-unsort-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 3 .csv',
        'hBM-1n-unsort-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 4 .csv',
        # 'hBM-2n-sort-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 5 .csv',
        # 'hBM-2n-sort-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 6 .csv',
        # 'hBM-2n-unsort-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 7 .csv',
        # 'hBM-2n-unsort-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 8 .csv',
        # 'hBM-3n-sort-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 9 .csv',
        # 'hBM-3n-sort-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 10 .csv',
        # 'hBM-4n-sort-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 11 .csv',
        # 'hBM-4n-sort-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 12 .csv',
        # 'hBM-5n-sort-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 13 .csv',
        # 'hBM-5n-sort-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 14 .csv',
    ]
    for p in pat_list:
        filename = p
        df = pd.read_csv(join(dirname(__file__), 'data/' + filename))
        ind = filename.split(".")[0]
        if 'Unnamed: 0' in df.columns:  # TODO drop all Unnamed
            df.drop(columns=['Unnamed: 0'], inplace=True)
        patients_data[ind] = df
        patient.options = patient.options + [ind]
        patient.value = ind
    upload_patients.button_type = 'success'

##################################################################### populations

    file_io = open(join(dirname(__file__), 'data/populations.txt'), "r")
    text = list(iter(file_io.read().splitlines()))
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

            df_viz.loc[indices, 'populationID'] = len(populations) - 1
            patches = {
                'populationID': [(i, len(populations) - 1) for i in indices]
            }
            source.patch(patches)
            bubble_name.value = ""

    upload_populations.button_type = 'success'
    layout.children[1] = draw_figure(df_viz, tree['edges'], populations)

##################################################################### clinical data

    clinical_data = pd.read_excel(join(dirname(__file__), 'data/PATIENTS DATABASE NEW FINAL to upload.xlsx'),
                                  header=[0, 1, 2])

    upload_clinical_data.button_type = 'success'
    groups_tabs.tabs[0] = create_panel()
    # groups[0][1] = manipulate_groups.map_measurements_to_patients(c_data=clinical_data,
    #                                                               pats_data=patients_data),

    add_group_button.disabled = False
    create_ref_group_button.disabled = False
    add_group()
    # print(groups)
    create_reference_group_tab()


load_test_data()

