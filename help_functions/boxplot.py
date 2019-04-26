import numpy as np
import pandas as pd
from bokeh.plotting import figure

# source: https://bokeh.pydata.org/en/latest/docs/gallery/boxplot.html


def create_boxplot(df=pd.DataFrame()):
    # generate some synthetic time series for six different categories
    if not df.empty:
        cats = df['group'].unique()
        value = df.columns[0]
        yy = np.random.randn(2000)
        g = np.random.choice(cats, 2000)
        for i, l in enumerate(cats):
            yy[g == l] += i // 2

        # find the quartiles and IQR for each category
        groups = df.groupby('group')
        q1 = groups.quantile(q=0.25)
        q2 = groups.quantile(q=0.5)
        q3 = groups.quantile(q=0.75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        lower = q1 - 1.5 * iqr

        # find the outliers for each category
        def outliers(group):
            cat = group.name
            return group[(group[value] > upper.loc[cat][value]) | (group[value] < lower.loc[cat][value])][value]

        out = groups.apply(outliers).dropna()

        # prepare outlier data for plotting, we need coordinates for every outlier.
        if not out.empty:
            outx = []
            outy = []
            for keys in out.index:
                outx.append(keys[0])
                outy.append(out.loc[keys[0]].loc[keys[1]])

        p = figure(tools="", background_fill_color="#efefef", x_range=cats, toolbar_location=None)

        # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
        qmin = groups.quantile(q=0.00)
        qmax = groups.quantile(q=1.00)
        upper[value] = [min([x, y]) for (x, y) in zip(list(qmax.loc[:, value]), upper[value])]
        lower[value] = [max([x, y]) for (x, y) in zip(list(qmin.loc[:, value]), lower[value])]

        # stems
        p.segment(cats, upper[value], cats, q3[value], line_color="black")
        p.segment(cats, lower[value], cats, q1[value], line_color="black")

        # boxes
        p.vbar(cats, 0.7, q2[value], q3[value], fill_color="#E08E79", line_color="black")
        p.vbar(cats, 0.7, q1[value], q2[value], fill_color="#3B8686", line_color="black")

        # whiskers (almost-0 height rects simpler than segments)
        # print(iqr)
        # print(iqr.iloc[:,0].max())
        # max_iqr = iqr.iloc[:,0].min()
        range_y = upper.iloc[:, 0].max() - lower.iloc[:, 0].min()
        p.rect(cats, lower[value], 0.2, range_y/500, line_color="black")
        p.rect(cats, upper[value], 0.2, range_y/500, line_color="black")

        # outliers
        if not out.empty:
            p.circle(outx, outy, size=6, color="#F38630", fill_alpha=0.6)

        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = "white"
        p.grid.grid_line_width = 2
        p.xaxis.major_label_text_font_size = "12pt"
    else:
        p = figure(tools="", background_fill_color="#efefef", x_range=['a','b','c'], toolbar_location=None)
    return p

