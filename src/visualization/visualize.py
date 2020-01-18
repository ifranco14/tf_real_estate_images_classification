import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import math
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
try:
    init_notebook_mode(connected=True)
except:
    pass
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def get_random_colors(n_colors, seed=32):
    colors = """
        aliceblue, antiquewhite, aqua, aquamarine, azure,
        beige, bisque, black, blanchedalmond, blue,
        blueviolet, brown, burlywood, cadetblue,
        chartreuse, chocolate, coral, cornflowerblue,
        cornsilk, crimson, cyan, darkblue, darkcyan,
        darkgoldenrod, darkgray, darkgrey, darkgreen,
        darkkhaki, darkmagenta, darkolivegreen, darkorange,
        darkorchid, darkred, darksalmon, darkseagreen,
        darkslateblue, darkslategray, darkslategrey,
        darkturquoise, darkviolet, deeppink, deepskyblue,
        dimgray, dimgrey, dodgerblue, firebrick,
        floralwhite, forestgreen, fuchsia, gainsboro,
        ghostwhite, gold, goldenrod, green,
        greenyellow, honeydew, hotpink, indianred, indigo,
        ivory, khaki, lavender, lavenderblush, lawngreen,
        lemonchiffon, lightblue, lightcoral, lightcyan,
        lightgoldenrodyellow, lightgray, lightgrey,
        lightgreen, lightpink, lightsalmon, lightseagreen,
        lightskyblue, lightslategray, lightslategrey,
        lightsteelblue, lightyellow, lime, limegreen,
        linen, magenta, maroon, mediumaquamarine,
        mediumblue, mediumorchid, mediumpurple,
        mediumseagreen, mediumslateblue, mediumspringgreen,
        mediumturquoise, mediumvioletred, midnightblue,
        mintcream, mistyrose, moccasin, navajowhite, navy,
        oldlace, olive, olivedrab, orange, orangered,
        orchid, palegoldenrod, palegreen, paleturquoise,
        palevioletred, papayawhip, peachpuff, peru, pink,
        plum, powderblue, purple, red, rosybrown,
        royalblue, saddlebrown, salmon, sandybrown,
        seagreen, sienna, silver, skyblue,
        slateblue, slategray, slategrey, springgreen,
        steelblue, tan, teal, thistle, tomato, turquoise,
        violet, wheat, yellow, yellowgreen
    """

    np.random.seed(seed)

    colors_list = colors.replace('\n', '').split(', ')
    random_colors = np.random.choice(colors_list, n_colors)

    print(random_colors)

    return random_colors


def plot_3d_scatter(datasets, features,
                    colors=['red', 'green', 'blue', 'yellow', 'orange'],
                    show_errors=True,
                    centroids=None):

    colors = [c for idx, c in enumerate(colors) if idx+1 <= len(datasets)]

    feature1, feature2, feature3 = features

    traces = []
    i = 0
    for dataset, c in zip(datasets, colors):
        dataset = dataset[dataset['y_true'] == dataset['y_pred']]
        x, y, z = dataset[feature1], dataset[feature2], dataset[feature3]
        trace = go.Scatter3d(
            name=f'group {i}',
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                color=c,
                size=12,
                line=dict(
                    color=c,
                    width=0.5
                ),
                opacity=0.3
            )
        )
        traces.append(trace)
        i += 1

    if show_errors:
        i = 0
        for dataset, c in zip(datasets, colors):
            dataset = dataset[dataset['y_true'] != dataset['y_pred']]
            x, y, z = dataset[feature1], dataset[feature2], dataset[feature3]
            errors = go.Scatter3d(
                name=f'Wrong predicted from group {i}',
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    color=c,
                    size=12,
                    line=dict(
                        color='black',
                        width=3
                    ),
                    opacity=1
                )
            )
            traces.append(errors)
            i += 1

    if centroids is not None:
        for (_, row), c in zip(centroids[::-1].iterrows(), colors):
            x, y, z = [row[feature1]], [row[feature2]], [row[feature3]]
            errors = go.Scatter3d(
                name='Centroid',
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    color='gray',
                    size=18,
                    line=dict(
                        color='black',
                        width=1.5
                    ),
                    opacity=0.8
                )
            )
            traces.append(errors)

    features = [f'{axis}: {f}' for axis, f in zip(['x', 'y', 'z'], features)]
    data = traces
    layout = go.Layout(
        title=go.layout.Title(
            text=f'3D scatter of {", ".join(features)}',
            xref='paper',
            x=0
        ),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=25
        )
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='simple-3d-scatter')


def plot_distribution_of_features_by_label(df, features=[],
                                            show_legend=False):
    '''
    Plots the distribution of the features by the y_true label, separating
    the true positives from the false positives cases.
    '''
    assert 'y_true' in df.columns, (
        'The provided dataframe must have `y_true` column')
    assert 'y_pred' in df.columns, (
        'The provided dataframe must have `y_pred` column')

    df['right_predicted'] = df['y_pred'] == df['y_true']

    if not features:
        features = [c for c in df.columns
                    if c not in
                    ['y_true', 'y_pred', 'right_predicted', 'probs',
                     'label']
                    ]

    n_rows = math.ceil(len(features) / 2)

    fig = make_subplots(rows=n_rows, cols=2,
                        subplot_titles=[f'Feature: {f.upper()}'
                                        for f in features])

    df_right_pred = df.loc[df.right_predicted]
    df_wrong_pred = df.loc[~df.right_predicted]

    hover_template = '<br>'.join([f'{col}'+'=%{customdata['+f'{idx}'+']}'
                                  for idx, col in enumerate(df.columns)])

    row_n = 1
    col_n = 1
    for idx, feature in enumerate(features):
        for label in df.y_true.unique():
            df_right_label = df_right_pred.loc[df_right_pred.label == label]
            fig.add_trace(get_box_plot_trace(
                y=df_right_label[feature],
                name=f'|{feature} - label: {label} - TP|',
                marker_color='green',
                box_mean=True,
                box_points='all', # can also be outliers/suspectedoutliers/False
                jitter=0.3,
                point_pos=-1.8,
                custom_data=df_right_label.values,
                hover_template=hover_template
            ), row=row_n, col=col_n)

            df_wrong_label = df_wrong_pred.loc[df_wrong_pred.label == label]
            fig.add_trace(get_box_plot_trace(
                y=df_wrong_label[feature],
                name=f'|{feature} - label: {label} - FP|',
                marker_color='red',
                box_mean=True,
                box_points='all', # can also be outliers/suspectedoutliers/False
                jitter=0.3,
                point_pos=-1.8,
                custom_data=df_wrong_label.values,
                hover_template=hover_template
            ), row=row_n, col=col_n)
        fig.update_xaxes(title_text="y_true", row=row_n, col=col_n)
        fig.update_yaxes(title_text=feature, row=row_n, col=col_n)

        if idx % 2 != 0:
            row_n += 1

        col_n = 1 if col_n == 2 else 2

    fig.update_layout(
        title=('Distribution of features: True Positives and False '
               +'Positives by ground truth label'),
        yaxis=dict(
            autorange=True,
            showgrid=True,
            zeroline=True,
            gridcolor='rgb(255, 255, 255)',
            gridwidth=1,
            zerolinecolor='rgb(255, 255, 255)',
            zerolinewidth=2,
        ),
        margin=dict(
            l=40,
            r=30,
            b=80,
            t=100,
        ),
        showlegend=show_legend
    )
    fig.show()


def get_box_plot_trace(y, name='', marker_color='red', box_mean=True,
                       box_points='all', jitter=.3, point_pos=-1.8,
                       custom_data=None, hover_template=None):
    """
    # box_points: can also be outliers/suspectedoutliers/False
    """
    return go.Box(
        y=y,
        name=name,
        marker_color=marker_color,
        boxmean=box_mean,
        boxpoints=box_points,
        jitter=jitter,
        pointpos=point_pos,
        customdata=custom_data,
        hovertemplate=hover_template
    )


def get_bar_trace(x, y, name):
    return go.Bar(name=name, x=x, y=y)


def get_series_trace(x, y, line_color='#388E3C', line_width=1, name='',
                     show_legend=False, fill=None, mode=None, opacity=1):
    return go.Scatter(name=name, x=x, y=y,
                      line=dict(color=line_color, width=line_width),
                      showlegend=show_legend, fill=fill, mode=mode,
                      opacity=opacity)


def get_series_with_std_traces(x, y, y_std, line_color='green', name='',
                               show_legend=False):
    line = get_series_trace(x=x, y=y, name=f'{name}',
                            line_color=line_color,
                            show_legend=True)

    line_m_std = get_series_trace(x=x, y=y-y_std, name=f'{name} - std',
                                  line_color=f'light{line_color}',
                                  line_width=.1, show_legend=False,
                                  fill='tonexty', opacity=.1, mode='lines')

    line_p_std = get_series_trace(x=x, y=y+y_std, name=f'{name} + std',
                                  line_color=line_color,
                                  line_width=.1, show_legend=False,
                                  fill='tonexty', opacity=.1, mode='lines')

    return [line, line_m_std, line_p_std]


def get_simple_layout(title, x_axis_name, y_axis_name):
    return {
        "title": {"text": f'{title}'},
        "xaxis": {"title": {"text": f"{x_axis_name}"}},
        "yaxis": {"title": {"text": f"{y_axis_name}"}},
        "legend": {
            "x": 0.8,
            "y": 0
        },
        "autosize": True
    }


def plot_figure(data, layout):
    fig = go.Figure(data=data, layout=layout)
    fig.show()
