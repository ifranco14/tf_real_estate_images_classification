import numpy as np

from bokeh.io import show, output_notebook
from bokeh.plotting import figure
from bokeh.core.properties import value
from bokeh.layouts import column
from bokeh.models import LinearColorMapper, BasicTicker, PrintfTickFormatter, ColorBar, ColumnDataSource, HoverTool
from bokeh.palettes import Category20
from bokeh.transform import cumsum, transform

TOOLTIPS = [('date', '@date{%F}'), ('COL', '@close')]

FORMATTERS = {
    'date': 'datetime', # use 'datetime' formatter for 'date' field
}

PLOT_HEIGHT = 750
PLOT_WIDTH = 900
toolbar_location = 'right'

def plot_bar(x, y, y_label=None, y_err=None,
             title=None, figsize=None, tooltips=None,
             multiple_plots=False, y_range_end=None, plot_height=None):
    '''
    Params:
        - x: x axis
        - y: y axis
        - y_label: label for y axis
        - title: title of the figure
    '''
    output_notebook()

    if plot_height is None:
        p_height = PLOT_HEIGHT
    else:
        p_height = plot_height

    if multiple_plots and len(x) > 100:
        s_ant = 0
        s_number = 1
        slices = np.arange(0, len(x), 100)
        for s in slices[1:]:
            source = ColumnDataSource(data=dict(x_axis=x[s_ant:s], y_axis=y[s_ant:s]))

            p = figure(x_range=x[s_ant:s],
                       plot_height=p_height,
                       plot_width=PLOT_WIDTH,
                       title=f'({s_number}/{len(slices)-1}) {title}',
                       toolbar_location=toolbar_location,
                       tooltips=tooltips)

            p.vbar(x='x_axis', top='y_axis', width=0.9,
                   color='#3288bd', source=source)

            if y_err is not None:
                p.segment(x[s_ant:s], y[s_ant:s]+y_err[s_ant:s], \
                          x[s_ant:s], y[s_ant:s]-y_err[s_ant:s], color="black")

            p.xgrid.grid_line_color = None
            p.y_range.start = 0

            if y_range_end is not None:
                p.y_range.end = y_range_end

            p.xaxis.major_label_orientation = np.pi/2

            show(p)

            s_ant = s
            s_number += 1
    else:
        source = ColumnDataSource(data=dict(x_axis=x, y_axis=y))

        p = figure(x_range=x,
                   plot_height=p_height,
                   plot_width=PLOT_WIDTH,
                   title=title,
                   toolbar_location=toolbar_location,
                   tooltips=tooltips)

        p.vbar(x='x_axis', top='y_axis', width=0.9,
               color='#3288bd', source=source)

        if y_err is not None:
            p.segment(x, y+y_err, x, y-y_err, color="black")

        p.xgrid.grid_line_color = None
        p.y_range.start = 0

        if y_range_end is not None:
            p.y_range.end = y_range_end

        p.xaxis.major_label_orientation = np.pi/2

        show(p)

def plot_stacked_bar(x, y, y_label=None, title=None):
    '''
    Params:
        - x: x axis (single array)
        - y: must be a list of time series indexed array
        - y_label: label for each y axis
        - title: title of the figure
    '''
    output_notebook()
    colors = ["#c9d9d3", "#718dbf", "#e84d60"]

    data = {y_axis_label: y_axis for y_axis, y_axis_label in zip(y, y_label)}
    data['x'] = x

    p = figure(x_range=x,
               plot_height=PLOT_HEIGHT,
               plot_width=PLOT_WIDTH,
               title=title,
               toolbar_location=toolbar_location)

    p.vbar_stack(y_label, x='x', width=0.9, color=colors, source=data,
                 legend=[value(x) for x in y_label])

    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.outline_line_color = None
    p.legend.location = 'top_right'
    p.legend.orientation = "vertical"
    p.legend.label_text_alpha = 0.8
    p.legend.background_fill_alpha = 0.3
    p.xaxis.major_label_orientation = np.pi/2

    show(p)

def plot_time_series(y, y_label=None, title=None,
                     legend_outside=False, tooltips=None,
                     formatters=None, mean=None, y_range_start=None):
    '''
    Params:
        - y: must be a list of time series indexed array
        - y_label: label for each y axis
        - title: title of the figure
    '''
    output_notebook()

    p = figure(plot_height=PLOT_HEIGHT-150,
               plot_width=PLOT_WIDTH,
               title=title,
               toolbar_location=toolbar_location,
               x_axis_type="datetime",
               x_axis_location="below",
               background_fill_color="#efefef")

    n_colors = len(y_label)

    if n_colors < 3:
        n_colors = 3

    colors = Category20[n_colors]
    i = 0
    for y_axis, y_axis_label in zip(y, y_label):
        dates = y_axis.index
        source = ColumnDataSource(data=dict(date=dates, close=y_axis))
        p.circle('date', 'close',
                 color=colors[i],
                 source=source, size=5)
        p.line('date', 'close',
               color=colors[i],
               source=source, legend=y_axis_label)
        i += 1

    p.legend.label_text_alpha = 0.8
    p.legend.background_fill_alpha = 0.3
    if y_range_start is not None:
        p.y_range.start = y_range_start

    p.add_tools(HoverTool(
        tooltips=tooltips,
        formatters=formatters,
        mode='mouse'
    ))

    if mean is not None:
        p.line(y[0].index,
               mean,
               legend=f'MEAN: {round(mean, 3)}',
               line_color="green", line_width=3, line_alpha=1)

    show(p)


def plot_pie(df,
             n_colors,
             legend_column,
             count_column=None,
             tooltips=None,
             title=''):

    if n_colors < 3:
        n_colors = 3

    d = df.copy()
    d['angle'] = d[count_column]/d[count_column].sum() * 2*np.pi
    d['color'] = Category20[n_colors][:d.shape[0]]
    d['percentage'] = d[count_column]/d[count_column].sum() * 100

    p = figure(plot_height=350, title=title, toolbar_location=None,
               tools="hover", tooltips=tooltips, x_range=(-0.5, 1.0))

    p.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend=legend_column, source=d)

    p.axis.axis_label=None
    p.axis.visible=False
    p.grid.grid_line_color = None

    show(p)


def box_plot(df, group_col, score_col, title=''):
    output_notebook()

    d = df.copy()

    d.loc[:, 'score'] = d[score_col]
    d.loc[:, group_col] = d[group_col].astype(str)
    boxes = d[group_col].unique()

    # find the quartiles and IQR for each category
    groups = d.groupby(group_col)
    q1 = groups.quantile(q=0.25)
    q2 = groups.quantile(q=0.5)
    q3 = groups.quantile(q=0.75)
    iqr = q3 - q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr

    # find the outliers for each category
    def outliers(group):
        cat = group.name
        return group[(group.score > upper.loc[cat]['score']) | (group.score < lower.loc[cat]['score'])]['score']
    out = groups.apply(outliers).dropna()

    # prepare outlier data for plotting, we need coordinates for every outlier.
    if not out.empty:
        outx = []
        outy = []
        for keys in out.index:
            outx.append(keys[0])
            outy.append(out.loc[keys[0]].loc[keys[1]])

    p = figure(title=title,
               plot_height=PLOT_HEIGHT-250,
               plot_width=PLOT_WIDTH,
               background_fill_color="#efefef",
               x_range=boxes,
               toolbar_location=toolbar_location)

    # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
    qmin = groups.quantile(q=0.00)
    qmax = groups.quantile(q=1.00)
    upper.score = [min([x,y]) for (x,y) in zip(list(qmax.loc[:,'score']),upper.score)]
    lower.score = [max([x,y]) for (x,y) in zip(list(qmin.loc[:,'score']),lower.score)]

    # stems
    p.segment(boxes, upper.score, boxes, q3.score, line_color="black")
    p.segment(boxes, lower.score, boxes, q1.score, line_color="black")

    # boxes
    p.vbar(boxes, 0.7, q2.score, q3.score, fill_color="#E08E79", line_color="black")
    p.vbar(boxes, 0.7, q1.score, q2.score, fill_color="#3B8686", line_color="black")

    # whiskers (almost-0 height rects simpler than segments)
    p.rect(boxes, lower.score, 0.2, 0.01, line_color="black")
    p.rect(boxes, upper.score, 0.2, 0.01, line_color="black")

    # outliers
    if not out.empty:
        p.circle(outx, outy, size=6, color="#F38630", fill_alpha=0.6)

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = "white"
    p.grid.grid_line_width = 2
    p.xaxis.major_label_text_font_size="12pt"

    show(p)


def plot_heat_map(data, tooltips=None, x_col=None, y_col=None, heat_col=None, title=None, low_cmap=None, high_cmap=None):
    output_notebook()

    if low_cmap is None:
        low_cmap = data[heat_col].min()

    if high_cmap is None:
        high_cmap = data[heat_col].max()

    source = ColumnDataSource(data)

    # this is the colormap from the original NYTimes plot
    colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
    mapper = LinearColorMapper(palette=colors,
                               low=low_cmap,
                               high=high_cmap)

    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

    x_range = data[x_col].astype(str).unique()
    y_range = list(reversed(data[y_col].unique()))
    data = data[~data.duplicated()]

    p = figure(title=title,
               plot_width=PLOT_WIDTH,
               plot_height=500,
               x_range=x_range,
               y_range=y_range,
               x_axis_location="above",
               tools=TOOLS,
               toolbar_location='below',
               tooltips=tooltips)

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "5pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = 3.14156 / 3

    p.rect(x=x_col, y=y_col, width=1, height=1, source=source,
       line_color=None, fill_color=transform(heat_col, mapper))


    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",
                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                         formatter=PrintfTickFormatter(format="%d"),
                         label_standoff=6, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')

    show(p)


def plot_scatter(df, feature1, feature2, title, tooltips=None, formatters=None, y_max=10, show=True):
    output_notebook()
    source = ColumnDataSource(df)

    p = figure(plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT, y_range=(0, y_max), title=title)

    p.circle(feature1, feature2, source=source, alpha=0.5, color='blue', legend=f'{feature1} vs {feature2}')

    p.add_tools(HoverTool(
                tooltips=tooltips,
                formatters=formatters,
                mode='mouse'
            ))

    p.legend.location = 'top_left'
    p.legend.label_text_alpha = 0.8
    p.legend.background_fill_alpha = 0.3

    show(p)
