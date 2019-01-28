import pandas as pd
import numpy as np
from bokeh.layouts import row, widgetbox
from bokeh.models import Select, ColorBar, ColumnDataSource, HoverTool, PointDrawTool, CustomJS
from bokeh.models.widgets import Button, Dropdown
from bokeh.models.mappers import LinearColorMapper
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
source = ColumnDataSource()


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
    if reduce(lambda a, b: a and b, [True if 'ok' in str[0] else False for str in dropdown.menu]):
        dropdown.button_type = "success"
        df_patients = hf.prepare_data(patient_data, coordinates)
        source = ColumnDataSource(df_patients)
        layout.children[1] = create_figure(df_patients)
        # print(df_patients.head())
        # print(edges.head())


def create_figure(patient_d):
    df = patient_d
    if not df.empty:
        xs = df[x.value].values
        ys = df[y.value].values
        x_title = x.value.title()
        y_title = y.value.title()

        kw = dict()
        # if x.value in patient.discrete:
        #     kw['x_range'] = sorted(set(xs))
        # if y.value in patient.discrete:
        #     kw['y_range'] = sorted(set(ys))
        kw['title'] = "%s vs %s" % (x_title, y_title)

        # tools = [hover] # 'pan,box_zoom,reset, wheel_zoom, box_select, lasso_select,tap'
        p = figure(plot_height=800, plot_width=1200,
                   tools='pan, box_zoom,reset, wheel_zoom, box_select, lasso_select,tap',
                   toolbar_location="above", **kw)

        p.xaxis.axis_label = x_title
        p.yaxis.axis_label = y_title

        # if x.value in patient.discrete:
        #     p.xaxis.major_label_orientation = pd.np.pi / 4

        if size.value != 'None':
            sizes = [hf.scale(value, df[size.value].min(),
                           df[size.value].max()) if not np.isnan(value) else 5 for value in df[size.value]]
        else:
            sizes = [9 for _ in df[x.value]]
        source.add(sizes, name='sz')

        if color.value != 'None':

            mapper = LinearColorMapper(palette=hf.create_color_map(),
                                  high=df[color.value].max(),
                                  high_color='red',
                                  low=df[color.value].min(),
                                  low_color='blue'
                                  )
            color_bar = ColorBar(color_mapper=mapper, location=(0, 0))


            # p.circle(x=xs, y=ys, color={'field': color.value,
            # 'transform': mapper}, size=sz, line_color="white", alpha=0.6,
            #          hover_color='white', hover_alpha=0.5)
            renderer = p.circle(x=x.value, y=y.value, color={'field': color.value, 'transform': mapper},
                     size='sz', line_color="white",
                     alpha=0.6, hover_color='white', hover_alpha=0.5, source=source)

            p.add_layout(color_bar, 'right')
        else:
            renderer = p.circle(x=x.value, y=y.value, size='sz', line_color="white", alpha=0.6,
                     hover_color='white', hover_alpha=0.5, source=source)

        # for line in range(0, edges.shape[0]):
        #     p.line([edges.loc[line, 'from_x'], edges.loc[line, 'to_x']],
        #     [edges.loc[line, 'from_y'], edges.loc[line, 'to_y']],
        #            line_width=0.5, color='white')

        hover = HoverTool(
            tooltips=[
                ("index", "$index"),
                ("cell count", "@{csv.pbm_036.count}"),
                ("(x,y)", "($x, $y)"),
                # ('close', '$@{adj close}{%0.2f}'),  # use @{ } for field names with spaces
                # ('volume', '@volume{0.00 a}'),
            ],
            renderers=[renderer]
            # formatters={
            #     'date': 'datetime',  # use 'datetime' formatter for 'date' field
            #     'adj close': 'printf',  # use 'printf' formatter for 'adj close' field
            # use default 'numeral' formatter for other fields
            # },
        )
        p.add_tools(hover)
        draw_tool = PointDrawTool(renderers=[renderer], empty_value='black')
        p.add_tools(draw_tool)
        p.toolbar.active_tap = draw_tool

        # plot styling
        p.title.text_color = "white"
        p.background_fill_color = "#1f1f1f"
        p.border_fill_color = "#2f2f2f"
        p.xaxis.major_label_text_color = "white"
        p.xaxis.axis_line_color = "white"
        p.xaxis.major_tick_line_color = "white"
        p.xaxis.minor_tick_line_color = "white"
        p.yaxis.major_label_text_color = "white"
        p.yaxis.axis_line_color = "white"
        p.yaxis.major_tick_line_color = "white"
        p.yaxis.minor_tick_line_color = "white"

        return p

    p = figure(plot_height=800, plot_width=1200,
               tools='pan, box_zoom,reset, wheel_zoom, box_select, lasso_select,tap',
               toolbar_location="above")
    return p


def update(attr, old, new):
    layout.children[1] = create_figure(df_patients)


# def update2(attr, old, new):
#     global source
#     if dataset.value == 'pat1':
#         source = ColumnDataSource(pat1.data)
#         layout.children[1] = create_figure(pat1)
#     else:
#         source = ColumnDataSource(pat2.data)
#         layout.children[1] = create_figure(pat2)

file_source.on_change('data', file_callback)

x = Select(title='X-Axis', value='x', options=df_patients.columns.tolist())
y = Select(title='Y-Axis', value='y', options=df_patients.columns.tolist())
size = Select(title='Size', value='None', options=['None'] + df_patients.columns.tolist())
color = Select(title='Color', value='None', options=['None'] + df_patients.columns.tolist())

y.on_change('value', update)
x.on_change('value', update)
size.on_change('value', update)
color.on_change('value', update)


menu = [("Upload patient data", "patient_data"), ("Upload cluster coordinates", "coordinates"),
        ("Upload graph edges", "edges")]
dropdown = Dropdown(label="Upload data", button_type="warning", menu=menu)
dropdown.callback = CustomJS(args=dict(file_source=file_source), code=up.file_read_callback)


# controls = widgetbox([x, y, color, size, dataset], width=200)
controls = widgetbox([dropdown, x, y, color, size], width=200)
layout = row(controls, create_figure(df_patients))


curdoc().add_root(layout)
curdoc().title = "Crossfilter"
