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


def create_reference_group_tab(group_no, groups_dict, remove_tab_func, rename_func, remove_measurements_func):

    remove_group_button = Button(label='remove group', width=100, button_type="danger")
    remove_group_button.on_click(remove_tab_func)
    group_name = TextInput(placeholder="rename group", width=150, css_classes=['renameGroupTextInput'])
    confirm = Button(label="OK", width=100)
    confirm.on_click(partial(rename_func, text_input=group_name))

    edit_box = widgetbox([remove_group_button, group_name, confirm], width=200, css_classes=['full-border'])

    categories = Div(text="""<h3>reference group</h3>""")

    remove_measurement = Button(label="remove measuremt(s)", button_type='danger', width=200)
    remove_measurement.on_click(remove_measurements_func)
    new_columns = [
        TableColumn(field='measurements', title='measurements'),
        TableColumn(field='patient', title='patient')
    ]
    patient_table = DataTable(source=ColumnDataSource(groups_dict[group_no][1]),
                              columns=new_columns, width=400, height=850, reorderable=True)

    new_tab = Panel(child=column(row(edit_box), categories,
                                 row(patient_table, remove_measurement)),
                    title="reference group")

    return new_tab


def create_panel(c_data, pats_data, rename_tab_func, remove_group_func,
                 remove_measurements_func, groups_dict, group_number):  # TODO css classes
    remove_group_button = Button(label='remove group', width=100, button_type="danger")
    remove_group_button.on_click(remove_group_func)
    group_name = TextInput(placeholder="rename group", width=150, css_classes=['renameGroupTextInput'])
    confirm = Button(label="OK", width=100)
    confirm.on_click(partial(rename_tab_func, text_input=group_name))

    edit_box = widgetbox([remove_group_button, group_name, confirm], width=200, css_classes=['full-border'])

    if c_data.empty:
        level_3 = PreText(text='please upload clinical data')
        new_tab = Panel(child=level_3, title="group " + str(group_number))

    else:
        # merge.options = merge.options + ["group " + str(group_number)]            # TODO merge
        level_1 = Select(title='category', value='None', width=200,
                         # css_classes=['select-width'],
                         options=['None'] + c_data.columns.get_level_values(0).unique().tolist())
        level_2 = Select(title='property', value='None', options=['None'], width=200)
        level_3 = PreText(text='please select an attribute')
        add_filter = Button(label='add condition', disabled=True, width=200)
        level_1.on_change('value', partial(select_columns, select_2=level_2, c_data=c_data))
        categories = Div(text="""""")
        remove_measurement = Button(label="remove measuremt(s)", button_type='danger', width=200)
        remove_measurement.on_click(remove_measurements_func)
        groups_dict[group_number][1] = map_measurements_to_patients(c_data, pats_data)
        # groups_dict[group_number][1].on_change('selected', enable_remove_button)
        new_columns = [
            TableColumn(field='measurements', title='measurements'),
            TableColumn(field='patient', title='patient')
        ]
        patient_table = DataTable(source=ColumnDataSource(groups_dict[group_number][1]),
                                  columns=new_columns, width=400, height=850, reorderable=True)
        # patient_table.source.selected.on_change('indices', partial(enable_remove_button, tab_no=group_number))
        filter_box = widgetbox([level_1, level_2, level_3, add_filter], css_classes=['full-border'])
        new_tab = Panel(child=column(row(filter_box, edit_box), categories,
                                     row(patient_table, remove_measurement)),
                        title="group " + str(group_number))
        level_2.on_change('value', partial(select_values, select_1=level_1, new_tab=new_tab, c_data=c_data))
        add_filter.on_click(partial(add_value_to_filter, new_tab=new_tab, group_no=group_number, c_data=c_data,
                                    groups_dict=groups_dict, pats_data=pats_data))

    return new_tab


def select_columns(attr, old, new, select_2, c_data):
    if new != 'None':
        select_2.options = c_data[new].columns.get_level_values(0).tolist() + ['None']
        select_2.value = 'None'
    else:
        select_2.options = ['None']
        select_2.value = 'None'


def select_values(attr, old, new, select_1, new_tab, c_data):
    if new != 'None':
        if c_data[select_1.value][new].values.dtype == 'object':  # categorical data
            level_3 = MultiSelect(title='value', value=['None'], options=['None'], width=200)
            try:
                # print("1  ", clinical_data[select_1.value][new].values.dtype)
                level_3.options = np.unique(c_data[select_1.value][new].iloc[:, 0].dropna().values).tolist()
                level_3.value = [level_3.options[0]]
            except TypeError:  # TODO filter non categorical data
                level_3.options = np.unique(
                    [str(obj) for obj in c_data[select_1.value][new].iloc[:, 0].dropna().values]).tolist()
            finally:
                new_tab.child.children[0].children[0].children[3].disabled = False
                new_tab.child.children[0].children[0].children[2] = column(level_3)

        elif 'datetime' in str(c_data[select_1.value][new].values.dtype):  # datetime data
            start = c_data[select_1.value][new].min().dt.date.item()
            end = c_data[select_1.value][new].max().dt.date.item()
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

        elif 'int' in str(c_data[select_1.value][new].values.dtype) or \
                'float' in str(c_data[select_1.value][new].values.dtype):
            # print("3   ", clinical_data[select_1.value][new].values.dtype)
            start = c_data[select_1.value][new].min().item()
            end = c_data[select_1.value][new].max().item()
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


def select_values_2(attr, old, new, w_box, c_data):
    if new != 'None':
        if c_data[w_box.children[1].value][new].values.dtype == 'object':  # categorical data
            level_3 = MultiSelect(title='value', value=['None'], options=['None'], width=180)
            try:
                # print("1  ", clinical_data[select_1.value][new].values.dtype)
                level_3.options = np.unique(c_data[w_box.children[1].value][new].iloc[:, 0].dropna().values).tolist()
                level_3.value = [level_3.options[0]]
            except TypeError:  # TODO filter non categorical data
                level_3.options = np.unique(
                    [str(obj) for obj in c_data[w_box.children[1].value][new].iloc[:, 0].dropna().values]).tolist()
            finally:
                w_box.children[3] = column(level_3)

        elif 'datetime' in str(c_data[w_box.children[1].value][new].values.dtype):  # datetime data
            start = c_data[w_box.children[1].value][new].min().dt.date.item()
            end = c_data[w_box.children[1].value][new].max().dt.date.item()
            date_slider = DateRangeSlider(title="",
                                          start=start,
                                          end=end,
                                          value=(start, end),
                                          # value_as_date=True,
                                          # step=1,
                                          width=180)
            checkbox_group = CheckboxGroup(labels=["invert selection"], active=[], width=180)
            w_box.children[3] = column(date_slider, checkbox_group)

        elif 'int' in str(c_data[w_box.children[1].value][new].values.dtype) or \
                'float' in str(c_data[w_box.children[1].value][new].values.dtype):
            # print("3   ", clinical_data[select_1.value][new].values.dtype)
            start = c_data[w_box.children[1].value][new].min().item()
            end = c_data[w_box.children[1].value][new].max().item()
            slider = RangeSlider(start=start, end=end, step=0.1, value=(start, end), title=new + " Range", width=180)
            checkbox_group = CheckboxGroup(labels=["invert selection"], active=[], width=180)
            w_box.children[3] = column(slider, checkbox_group)

        else:
            print("Something went wrong, unexpected datatype by clinical data value selecting")  # TODO error message?

    else:
        w_box.children[3] = PreText(text='please select a property', width=200)


def add_value_to_filter(new_tab, group_no, c_data, groups_dict, pats_data):
    level_1 = new_tab.child.children[0].children[0].children[0].value
    level_2 = new_tab.child.children[0].children[0].children[1].value
    level_3 = new_tab.child.children[0].children[0].children[2].children
    # group_no = groups_tabs.active
    # print("old", groups)
    df = c_data[level_1][level_2]
    if len(level_3) == 2:
        invert = len(level_3[1].active) == 1

        if 'datetime' in str(c_data[level_1][level_2].values.dtype):
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

        if level_1 not in groups_dict[group_no][0].keys():
            groups_dict[group_no][0][level_1] = {}
        groups_dict[group_no][0][level_1][level_2] = (invert, start, end)

    else:
        categories = level_3[0].value
        i = df[df.columns[0]][df[df.columns[0]].isin(categories)].index
        if level_1 in groups_dict[group_no][0].keys():
            groups_dict[group_no][0][level_1][level_2] = categories
        else:
            groups_dict[group_no][0][level_1] = {}
            groups_dict[group_no][0][level_1][level_2] = categories

    measurements = find_measurements(i, pats_data)
    new_df = pd.DataFrame(columns=['measurements', 'patient'])
    for k, v in measurements.items():
        for val in v:
            df = pd.DataFrame([[val, k]], columns=['measurements', 'patient'])
            new_df = new_df.append(df)
    groups_dict[group_no][1] = new_df.reset_index(drop=True)
    # print("GGG", new_df)
    # print("#######################################################")
    new_tab.child.children[2].children[0].source = ColumnDataSource(groups_dict[group_no][1])

    new_tab.child.children[1].text = write_conditions(groups_dict[group_no][0], groups_dict[group_no][1].shape[0])
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


def map_measurements_to_patients(c_data, pats_data):
    # print("#############################################################################")
    new_df = pd.DataFrame(columns=['measurements', 'patient'])
    pat_list = c_data.index.dropna()
    # measurement_list = patients_data.keys()
    for k, v in find_measurements(pat_list, pats_data).items():
        for val in v:
            df = pd.DataFrame([[val, k]], columns=['measurements', 'patient'])
            new_df = new_df.append(df)
    # print(new_df)
    # print("#############################################################################")
    return new_df.reset_index(drop=True)


def find_measurements(patient_list, pats_data, output='dict'):
    # print(old_list)
    # print(patient_list)
    if output == 'dict':
        measurement_list = {}
        for pat in patient_list:
            measurement_list[pat] = []
            for measurement in pats_data.keys():
                if pat + "-" in measurement:
                    measurement_list[pat].append(measurement)
            if not measurement_list[pat]:
                measurement_list.pop(pat)
        return measurement_list
    else:
        measurement_list = []
        for pat in patient_list:
            for measurement in pats_data.keys():
                if pat + "-" in measurement:
                    measurement_list.append(measurement)
        return measurement_list
