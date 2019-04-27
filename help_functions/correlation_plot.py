from bokeh.plotting import figure
from math import pi
import pandas as pd
import numpy as np
import help_functions.help_functions as hf
from bokeh.models.mappers import LinearColorMapper
from bokeh.models import ColorBar


def correlation_plot(marker_value, pats_data):
    if marker_value != "None":
        patients_list = list(pats_data.keys())

        p = figure(title="correlation on marker " + str(marker_value),
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

        df = pd.DataFrame(index=pats_data[patients_list[0]].index.copy())

        for pat in patients_list:
            df[pat] = pats_data[pat][marker_value] if marker_value in pats_data[pat].columns else np.NaN

        df.columns = patients_list

        df = pd.DataFrame(df.corr().stack(), columns=['rate']).reset_index()

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
