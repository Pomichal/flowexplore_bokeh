import numpy as np
from bokeh.models import ColorBar, HoverTool, PointDrawTool, LassoSelectTool
from bokeh.models.widgets import TableColumn, NumberFormatter
from bokeh.models.mappers import LinearColorMapper
from bokeh.plotting import figure

import help_functions.help_functions as hf

formatter = NumberFormatter(format='0.0000')


def create_figure(df, df_edges, df_populations, source, x_value, y_value, color_value, size_value):

    new_columns = []

    if not df.empty:

        pop_names = [df_populations.iloc[pop_id]['population_name'] if pop_id != -1 else '???'
                     for pop_id in df['populationID']]
        source.add(pop_names, name='pop_names')

        x_title = x_value.title()
        y_title = y_value.title()

        kw = dict()
        kw['title'] = "%s vs %s" % (x_title, y_title)

        p = figure(plot_height=900, plot_width=1200,
                   tools='pan, box_zoom,reset, wheel_zoom, box_select, tap, save',
                   toolbar_location="above", **kw)
        p.add_tools(LassoSelectTool(select_every_mousemove=False))

        p.xaxis.axis_label = x_title
        p.yaxis.axis_label = y_title

        # add lines
        if not df_edges.empty:
            lines_from = []
            lines_to = []
            for line in range(0, df_edges.shape[0]):
                lines_from.append(
                    [source.data[x_value][df_edges.iloc[line, 1] - 1],  # TODO filter possible nan values
                     source.data[x_value][df_edges.iloc[line, 2] - 1]])
                lines_to.append([source.data[y_value][df_edges.iloc[line, 1] - 1],  # TODO filter possible nan values
                                 source.data[y_value][df_edges.iloc[line, 2] - 1]])

            p.multi_line(lines_from, lines_to, line_width=0.5, color='white')

        # mark populations
        line_color = ['white'] * len(df)
        line_width = [1] * len(df)
        if not df_populations.empty:
            line_color = [df_populations.iloc[pop_id]['color'] if pop_id != -1 else 'white'
                          for pop_id in df['populationID']]
            line_width = [5 if lc != 'white' else 1 for lc in line_color]

        source.add(line_color, name='lc')
        source.add(line_width, name='lw')

        if size_value != 'None':
            sizes = [hf.scale(value, df[size_value].min(),
                              df[size_value].max()) if not np.isnan(value) and value != 0 else 10 for value in
                     df[size_value]]
        else:
            sizes = [25 for _ in df[x_value]]
        source.add(sizes, name='sz')

        if color_value != 'None':
            mapper = LinearColorMapper(palette=hf.create_color_map(),
                                       high=df[color_value].max(),
                                       high_color='red',
                                       low=df[color_value].min(),
                                       low_color='blue'
                                       )
            color_bar = ColorBar(color_mapper=mapper, location=(0, 0))

            renderer = p.circle(x=x_value, y=y_value, color={'field': color_value, 'transform': mapper},
                                size='sz',
                                line_color="lc",
                                line_width="lw",
                                line_alpha=0.9,
                                alpha=0.6, hover_color='white', hover_alpha=0.5, source=source)
            p.add_layout(color_bar, 'right')

        else:
            renderer = p.circle(x=x_value, y=y_value, size='sz',
                                line_color="lc",
                                line_width="lw",
                                line_alpha=0.9,
                                alpha=0.6,
                                hover_color='white', hover_alpha=0.5,
                                source=source)

        hover = HoverTool(
            tooltips=[
                ("index", "$index"),
                ("{}".format(size_value), "@{{{}}}".format(size_value)),
                ("{}".format(color_value), "@{{{}}}".format(color_value)),
                ("population", "@pop_names"),
                ("(x,y)", "($x, $y)"),
            ],
            renderers=[renderer]
        )

        p.add_tools(hover)
        draw_tool = PointDrawTool(renderers=[renderer], add=False)
        p.add_tools(draw_tool)
        p.toolbar.active_tap = draw_tool

        new_columns = [
            # TableColumn(field=x_value, title=x_value, formatter=formatter),
            # TableColumn(field=y_value, title=y_value, formatter=formatter),
            # TableColumn(field=color_value, title=color_value, formatter=formatter),
            # TableColumn(field=size_value, title=size_value, formatter=formatter),
            TableColumn(field=x_value, title=x_value),
            TableColumn(field=y_value, title=y_value),
            TableColumn(field=color_value, title=color_value),
            TableColumn(field=size_value, title=size_value),
            TableColumn(field='pop_names', title="population"),
        ]
        return p, new_columns

    p = figure(plot_height=900, plot_width=1200,
               tools='pan, box_zoom,reset, wheel_zoom, box_select, lasso_select,tap',
               toolbar_location="above")
    return p, new_columns
