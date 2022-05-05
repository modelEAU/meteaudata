from copy import deepcopy
from typing import Dict, Literal, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go

from filters.utilities import (align_results_in_time,
                               apply_observations_to_outliers)


def get_default_plot_elements(
    template: Literal["presentation", "plotly_white"]
) -> go.Layout:
    return go.Layout(
        template=template,
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
    template: Literal["presentation", "plotly_white"] = "plotly_white",
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
    layout = get_default_plot_elements(template=template)
    fig.update_layout(layout)
    return fig


def get_clean_column_names(
    language: Literal["french", "english"] = "english"
) -> Dict[str, str]:
    names = {
        "outlier_value": {
            "english": "Outliers",
            "french": "Aberrant",
        },
        "input_is_outlier": {
            "english": "Outlier Detected",
            "french": "Valeur Aberrante Détectée",
        },
        "input_value": {
            "english": "Observed",
            "french": "Observations",
        },
        "accepted_value": {
            "english": "Accepted",
            "french": "Filtré",
        },
        "predicted_value": {
            "english": "Predicted",
            "french": "Prédit",
        },
        "predicted_upper_limit": {
            "english": "Upper Bound",
            "french": "Limite Supérieure",
        },
        "predicted_lower_limit": {
            "english": "Lower Bound",
            "french": "Limite inférieure",
        },
        "smoothed": {
            "english": "Smoothed",
            "french": "Lissé",
        },
    }
    return {x: names[x][language] for x in names}


def plot_results(
    df: pd.DataFrame,
    series_name: str,
    template: Literal["presentation", "plotly_white"] = "plotly_white",
    language: Literal["french", "english"] = "english",
) -> go.Figure:
    df = align_results_in_time(df)
    df = apply_observations_to_outliers(df)
    fig = go.Figure()
    fig.update_layout(get_default_plot_elements(template))
    x = df["date"]
    names = get_clean_column_names(language)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["predicted_lower_limit"],
            name=names["predicted_lower_limit"],
            line=dict(width=0),
            mode="lines",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["predicted_upper_limit"],
            name=names["predicted_upper_limit"],
            line=dict(width=0),
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(205, 229, 203, 0.5)",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["input_value"],
            name=names["input_value"],
            line=dict(color="black"),
            mode="lines+markers",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["predicted_value"],
            name=names["predicted_value"],
            line=dict(color="green", dash="dot"),
            mode="lines+markers",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["outlier_value"],
            name=names["outlier_value"],
            line=dict(color="black"),
            mode="markers",
            marker=dict(symbol="x", size=15),
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["accepted_value"],
            name=names["accepted_value"],
            line=dict(color="#388E3C", dash="dash"),
            mode="lines",
            showlegend=True,
        )
    )
    if "Smoothed" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["smoothed"],
                name=names["smoothed"],
                line=dict(color="#388E3C"),
                mode="lines",
                showlegend=True,
            )
        )
    title_start = (
        "Filtration results" if language == "english" else "Résultats de filtration"
    )
    fig.update_layout(
        title=f"{title_start}: {series_name}",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Value" if language == "english" else "Valeur"),
    )
    return fig
