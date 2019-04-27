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
from help_functions import create_figure, manipulate_figure

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
    groups[0][1] = map_measurements_to_patients()

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
        x.options = df_viz.columns.tolist()
        x.value = df_viz.columns.tolist()[0]
        y.options = df_viz.columns.tolist()
        y.value = df_viz.columns.tolist()[1]
        color.options = ['None'] + df_viz.columns.tolist()
        color.value = 'None'
        size.options = ['None'] + df_viz.columns.tolist()
        size.value = 'None'
    layout.children[1] = draw_figure(df_viz, tree['edges'], populations)


def check_selection(attr, old, new):
    IDs = set(list(source.data['populationID'][k] for k in source.selected.indices))
    if len(IDs) == 1:
        bubble_select.disabled = False
    else:
        bubble_select.disabled = True


def create_stats_tables():  # TODO remove error, if running without coordinates data
    global df_stats
    # create empty dataframe with multiindex
    b = populations['population_name'].values
    m = reduce(lambda s, q: s | q, map(lambda p: set(patients_data[p].columns.values), patients_data.keys()))
    iterables = [b, m]
    # print(b, m)

    df_stats = pd.DataFrame(index=pd.MultiIndex.from_product(iterables), columns=patients_data.keys())
    # print(df_stats)
    marker.options = ['None'] + list(m)

    cell_count_column = 'None'

    for pat in patients_data:
        df_patient = patients_data[pat]

        # select column with cell count
        # TODO better condition to filter columns and find cell count
        int_cols = list(filter(
            lambda a: (not "id" in a.lower()) and df_patient[a].nunique() > 2 and ('int' in str(df_patient[a].values.dtype)),
            df_patient.columns))
        # print(pat, int_cols)
        if len(int_cols) != 1:
            print("ERROR")  # TODO error message
            return
        else:
            cell_count_column = int_cols[0]
            # print("THIS IS", cell_count_column)
            cell_sum = df_patient[cell_count_column].sum()
            # print("###############################################", m)
            # print(b)
            # print(cell_sum)
            for idx_b, bu in enumerate(b):
                # print(idx_b, bu)
                clusters = df_patient[df_viz['populationID'] == idx_b]
                for idx_m, ma in enumerate(m):
                    if ma != cell_count_column and ma in clusters.columns:
                        values = map(lambda a, count: a * clusters.loc[count, cell_count_column],
                                     clusters[ma].dropna(), clusters[ma].index.values)
                        df_stats.loc[(bu, ma), pat] = reduce(lambda p, q: p + q, list(values), 0) / cell_sum
                    else:
                        df_stats.loc[(bu, ma), pat] = reduce(lambda p, q: p + q, clusters[cell_count_column].values)
    df_stats = df_stats.astype(float)
    # print(df_stats)
    marker.value = cell_count_column
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
        # print(df)

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


def find_healthy():
    # print("#############################################################################")
    measurements = []
    for pat in patients_data.keys():
        pat_name = "-".join(pat.split("-")[:2])
        if pat_name not in clinical_data.index:
            measurements += [pat]

    new_df = pd.DataFrame(columns=['measurements', 'patient'])

    for m in measurements:
        df = pd.DataFrame([[m, 'healthy']], columns=['measurements', 'patient'])
        new_df = new_df.append(df)
    return new_df.reset_index(drop=True)


def create_reference_group_tab():
    group_number = len(groups_tabs.tabs)

    remove_group_button = Button(label='remove group', width=100, button_type="danger")
    remove_group_button.on_click(remove_group)
    group_name = TextInput(placeholder="rename group", width=150, css_classes=['renameGroupTextInput'])
    confirm = Button(label="OK", width=100)
    confirm.on_click(partial(rename_tab, text_input=group_name))

    edit_box = widgetbox([remove_group_button, group_name, confirm], width=200, css_classes=['full-border'])

    categories = Div(text="""<h3>reference group</h3>""")

    remove_measurement = Button(label="remove measuremt(s)", button_type='danger', width=200)
    remove_measurement.on_click(remove_measurements)
    groups.append([{}, pd.DataFrame(map_measurements_to_patients(), columns=['measurements', 'patient'])])
    groups[group_number][1] = find_healthy()
    # groups[group_number][1].on_change('selected', enable_remove_button)
    new_columns = [
        TableColumn(field='measurements', title='measurements'),
        TableColumn(field='patient', title='patient')
    ]
    patient_table = DataTable(source=ColumnDataSource(groups[group_number][1]),
                              columns=new_columns, width=400, height=850, reorderable=True)
    # new_tab = Panel(child=row(patient_table, remove_measurement), title="reference group")

    new_tab = Panel(child=column(row(edit_box), categories,
                                 row(patient_table, remove_measurement)),
                    title="reference group")

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
    groups.append([{}, pd.DataFrame(map_measurements_to_patients(), columns=['measurements', 'patient'])])
    groups_tabs.tabs = groups_tabs.tabs + [create_panel(group_number)]
    groups_tabs.active = len(groups_tabs.tabs) - 1


def remove_group():
    global merge
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
    remove_group_button = Button(label='remove group', width=100, button_type="danger")
    remove_group_button.on_click(remove_group)
    group_name = TextInput(placeholder="rename group", width=150, css_classes=['renameGroupTextInput'])
    confirm = Button(label="OK", width=100)
    confirm.on_click(partial(rename_tab, text_input=group_name))

    edit_box = widgetbox([remove_group_button, group_name, confirm], width=200, css_classes=['full-border'])

    if clinical_data.empty:
        level_3 = PreText(text='please upload clinical data')
        new_tab = Panel(child=level_3, title="group " + str(group_number))

    else:
        # merge.options = merge.options + ["group " + str(group_number)]            # TODO merge
        level_1 = Select(title='category', value='None', width=200,
                         # css_classes=['select-width'],
                         options=['None'] + clinical_data.columns.get_level_values(0).unique().tolist())
        level_2 = Select(title='property', value='None', options=['None'], width=200)
        level_3 = PreText(text='please select an attribute')
        add_filter = Button(label='add condition', disabled=True, width=200)
        level_1.on_change('value', partial(select_columns, select_2=level_2))
        categories = Div(text="""""")
        remove_measurement = Button(label="remove measuremt(s)", button_type='danger', width=200)
        remove_measurement.on_click(remove_measurements)
        groups[group_number][1] = map_measurements_to_patients()
        # groups[group_number][1].on_change('selected', enable_remove_button)
        new_columns = [
            TableColumn(field='measurements', title='measurements'),
            TableColumn(field='patient', title='patient')
        ]
        patient_table = DataTable(source=ColumnDataSource(groups[group_number][1]),
                                  columns=new_columns, width=400, height=850, reorderable=True)
        # patient_table.source.selected.on_change('indices', partial(enable_remove_button, tab_no=group_number))
        filter_box = widgetbox([level_1, level_2, level_3, add_filter], css_classes=['full-border'])
        new_tab = Panel(child=column(row(filter_box, edit_box), categories,
                                     row(patient_table, remove_measurement)),
                        title="group " + str(group_number))
        level_2.on_change('value', partial(select_values, select_1=level_1, new_tab=new_tab))
        add_filter.on_click(partial(add_value_to_filter, new_tab=new_tab))

    return new_tab


def select_columns(attr, old, new, select_2):
    if new != 'None':
        select_2.options = clinical_data[new].columns.get_level_values(0).tolist() + ['None']
        select_2.value = 'None'
    else:
        select_2.options = ['None']
        select_2.value = 'None'


def select_values(attr, old, new, select_1, new_tab):
    if new != 'None':
        if clinical_data[select_1.value][new].values.dtype == 'object':  # categorical data
            level_3 = MultiSelect(title='value', value=['None'], options=['None'], width=200)
            try:
                # print("1  ", clinical_data[select_1.value][new].values.dtype)
                level_3.options = np.unique(clinical_data[select_1.value][new].iloc[:, 0].dropna().values).tolist()
                level_3.value = [level_3.options[0]]
            except TypeError:  # TODO filter non categorical data
                level_3.options = np.unique(
                    [str(obj) for obj in clinical_data[select_1.value][new].iloc[:, 0].dropna().values]).tolist()
            finally:
                new_tab.child.children[0].children[0].children[3].disabled = False
                new_tab.child.children[0].children[0].children[2] = column(level_3)

        elif 'datetime' in str(clinical_data[select_1.value][new].values.dtype):  # datetime data
            start = clinical_data[select_1.value][new].min().dt.date.item()
            end = clinical_data[select_1.value][new].max().dt.date.item()
            date_slider = DateRangeSlider(title="",
                                          start=start,
                                          end=end,
                                          value=(start, end),
                                          # value_as_date=True,
                                          # step=1,
                                          width=180)
            checkbox_group = CheckboxGroup(labels=["invert selection"], active=[], width=180)
            new_tab.child.children[0].children[0].children[3].disabled = False
            new_tab.child.children[0].children[0].children[2] = column(date_slider, checkbox_group)

        elif 'int' in str(clinical_data[select_1.value][new].values.dtype) or \
                'float' in str(clinical_data[select_1.value][new].values.dtype):
            # print("3   ", clinical_data[select_1.value][new].values.dtype)
            start = clinical_data[select_1.value][new].min().item()
            end = clinical_data[select_1.value][new].max().item()
            slider = RangeSlider(start=start, end=end, step=0.1, value=(start, end), title=new + " Range", width=180)
            checkbox_group = CheckboxGroup(labels=["invert selection"], active=[], width=180)
            new_tab.child.children[0].children[0].children[3].disabled = False
            new_tab.child.children[0].children[0].children[2] = column(slider, checkbox_group)

        else:
            print("Something went wrong, unexpected datatype by clinical data value selecting")  # TODO error message?

    else:
        new_tab.child.children[0].children[0].children[2] = \
            PreText(text='please select an attribute', width=200)
        new_tab.child.children[0].children[0].children[3].disabled = True


def find_measurements(patient_list):
    # print(old_list)
    # print(patient_list)
    measurement_list = {}
    for pat in patient_list:
        measurement_list[pat] = []
        for measurement in patients_data.keys():
            if pat + "-" in measurement:
                measurement_list[pat].append(measurement)
        if not measurement_list[pat]:
            measurement_list.pop(pat)
    return measurement_list


def add_value_to_filter(new_tab):
    level_1 = new_tab.child.children[0].children[0].children[0].value
    level_2 = new_tab.child.children[0].children[0].children[1].value
    level_3 = new_tab.child.children[0].children[0].children[2].children
    group_no = groups_tabs.active
    # print("old", groups)
    df = clinical_data[level_1][level_2]
    if len(level_3) == 2:
        invert = len(level_3[1].active) == 1

        if 'datetime' in str(clinical_data[level_1][level_2].values.dtype):
            start = level_3[0].value_as_date[0]
            end = level_3[0].value_as_date[1]
            if invert:
                i = df[df.columns[0]][(df[df.columns[0]] < pd.Timestamp(start))
                                      | (df[df.columns[0]] > pd.Timestamp(end))].index
            else:
                i = df[df.columns[0]][(df[df.columns[0]] > pd.Timestamp(start))
                                      & (df[df.columns[0]] < pd.Timestamp(end))].index
        else:  # if number
            start = level_3[0].value[0]
            end = level_3[0].value[1]
            if invert:
                i = df[df.columns[0]][(df[df.columns[0]] < start)
                                      | (df[df.columns[0]] > end)].index
            else:
                i = df[df.columns[0]][(df[df.columns[0]] > start)
                                      & (df[df.columns[0]] < end)].index

        if level_1 not in groups[group_no][0].keys():
            groups[group_no][0][level_1] = {}
        groups[group_no][0][level_1][level_2] = (invert, start, end)

    else:
        categories = level_3[0].value
        i = df[df.columns[0]][df[df.columns[0]].isin(categories)].index
        if level_1 in groups[group_no][0].keys():
            groups[group_no][0][level_1][level_2] = categories
        else:
            groups[group_no][0][level_1] = {}
            groups[group_no][0][level_1][level_2] = categories

    measurements = find_measurements(i)
    new_df = pd.DataFrame(columns=['measurements', 'patient'])
    for k, v in measurements.items():
        for val in v:
            df = pd.DataFrame([[val, k]], columns=['measurements', 'patient'])
            new_df = new_df.append(df)
    groups[group_no][1] = new_df.reset_index(drop=True)
    # print("GGG", new_df)
    # print("#######################################################")
    new_tab.child.children[2].children[0].source = ColumnDataSource(groups[group_no][1])

    new_tab.child.children[1].text = write_conditions(groups[group_no][0], groups[group_no][1].shape[0])
    # print("new", groups)
    # print(write_conditions(groups[group_no]))
    # print()


def write_conditions(conditions, group_size, tag="li"):  # TODO drop empty conditions
    # conditions_text = "<h4>Group size: " + str(group_size) + "</h4><h3>conditions:</h3><ul>"
    conditions_text = "<h3>conditions:</h3><ul>"

    for k, v in conditions.items():
        for key, value in v.items():
            if len(value) == 3 and type(value[0]) is bool:
                if value[0]:
                    conditions_text += "<" + tag + ">" + key + " in &lt; min, " + str(value[1]) + "&gt; OR &lt; " \
                                       + str(value[2]) + ", max &gt; </" + tag + ">"
                else:
                    conditions_text += "<" + tag + ">" + key + " in &lt; " + str(value[1]) + " , " + str(value[2]) \
                                       + " &gt; </" + tag + ">"
            else:
                conditions_text += "<" + tag + ">" + key + " in " + str(value) + "</" + tag + ">"

    return conditions_text + "</ul>"


def map_measurements_to_patients():
    # print("#############################################################################")
    new_df = pd.DataFrame(columns=['measurements', 'patient'])
    pat_list = clinical_data.index.dropna()
    # measurement_list = patients_data.keys()
    for k, v in find_measurements(pat_list).items():
        for val in v:
            df = pd.DataFrame([[val, k]], columns=['measurements', 'patient'])
            new_df = new_df.append(df)
    # print(new_df)
    # print("#############################################################################")
    return new_df.reset_index(drop=True)


def active_tab(attr, old, new):
    if old == 0 and new == 1:
        create_stats_tables()
    if new == 2:
        stats_df = pd.DataFrame(df_stats.loc[df_stats.index[0], :])
        # print(df_stats.index.get_level_values(0).unique().tolist())
        # print(df_stats.index.get_level_values(1).unique().tolist())
        bubbles.options = ["None"] + df_stats.index.get_level_values(0).unique().tolist()
        markers.options = ["None"] + df_stats.index.get_level_values(1).unique().tolist()
        # print()
        bubbles.value = bubbles.options[1]
        markers.value = markers.options[1]
        # for i in stats_df.index:
        #     for group_number, group in enumerate(groups):
        #         if i in group[1]['measurements'].tolist():
        #             stats_df.loc[i, 'group'] = groups_tabs.tabs[group_number].title
        # layout3.children[1] = boxplot.create_boxplot(stats_df)


def draw_boxplot(attr, old, new):
    b = bubbles.value
    m = markers.value
    if b != 'None' and m != 'None':
        stats_df = pd.DataFrame(df_stats.loc[(b, m), :])
        # print(stats_df)
        for i in stats_df.index:
            for group_number, group in enumerate(groups):
                if i in group[1]['measurements'].tolist():
                    stats_df.loc[i, 'group'] = groups_tabs.tabs[group_number].title
        # print(stats_df)
        layout3.children[1] = boxplot.create_boxplot(stats_df)


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
formatter = NumberFormatter(format='0.0000')
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
groups.append([{}, pd.DataFrame(map_measurements_to_patients(), columns=['measurements', 'patient'])])

groups_tabs = Tabs(tabs=[group1])
groups_tabs.width = 800

layout2 = row(basic_overview, correlation_plot(), column(children=[column(row(add_group_button,
                                                                              create_ref_group_button,
                                                                              upload_clinical_data),
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

layout3 = row(column(bubbles, markers), boxplot.create_boxplot())

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
        'pBM-39-unsort-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 193 .csv',
        'pBM-39-unsort-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 194 .csv',
        'pBM-41-sort-CD15-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 195 .csv',
        'pBM-41-sort-CD15-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 196 .csv',
        'pBM-43-sort-CD15-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 197 .csv',
        'pBM-43-sort-CD15-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 198 .csv',
        'pBM-48-sort-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 199 .csv',
        'pBM-48-sort-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 200 .csv',
        'pBM-49-sort-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 201 .csv',
        'pBM-49-sort-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 202 .csv',
        'hBM-1n-sort-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 1 .csv',
        'hBM-1n-sort-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 2 .csv',
        'hBM-1n-unsort-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 3 .csv',
        'hBM-1n-unsort-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 4 .csv',
        'hBM-2n-sort-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 5 .csv',
        'hBM-2n-sort-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 6 .csv',
        'hBM-2n-unsort-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 7 .csv',
        'hBM-2n-unsort-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 8 .csv',
        'hBM-3n-sort-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 9 .csv',
        'hBM-3n-sort-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 10 .csv',
        'hBM-4n-sort-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 11 .csv',
        'hBM-4n-sort-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 12 .csv',
        'hBM-5n-sort-p2-concat-normalized-noCD3-CD14-CD15.fcs _ 13 .csv',
        'hBM-5n-sort-p3-concat-normalized-noCD3-CD14-CD15.fcs _ 14 .csv',
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
    groups[0][1] = map_measurements_to_patients()

    add_group_button.disabled = False
    create_ref_group_button.disabled = False
    add_group()
    create_reference_group_tab()


load_test_data()

