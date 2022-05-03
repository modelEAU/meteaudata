from copy import deepcopy
from typing import Literal, Optional, Union

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go


def get_default_plot_elements(
    mode: Union[Literal["presentation"], Literal["plotly_white"]]
) -> go.Layout:
    return go.Layout(
        template=mode,
        hovermode="x unified",  # To compare on hover
        legend=dict(yanchor="top", xanchor="left", orientation="h", y=1.05, x=0),
    )


def add_traces_to_other_plot(
    origin_plot: go.Figure, final_plot: go.Figure
) -> go.Figure:
    final_plot = deepcopy(final_plot)
    traces_to_add = origin_plot.data
    for trace in traces_to_add:
        final_plot.add_trace(trace)
    return final_plot


def plot_array(
    data: npt.NDArray,
    series_name: str = None,
    plot_title: Optional[str] = None,
    mode: Union[Literal["presentation"], Literal["plotly_white"]] = "plotly_white",
) -> go.Figure:
    fig = go.Figure()
    data = data.reshape(
        -1,
    )
    scatter = go.Scatter(
        x=np.linspace(0, len(data), num=len(data)),
        y=data,
        name=series_name,
        mode="lines+markers",
    )
    fig.add_trace(scatter)
    layout = get_default_plot_elements(mode=mode)
    fig.update_layout(layout)
    return fig
