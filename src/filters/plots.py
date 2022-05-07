from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from filters.utilities import (align_results_in_time,
                               apply_observations_to_outliers)

pio.templates.default = "plotly_white"

CLEAN_NAMES = {
    "outlier_values": {
        "english": "Outliers",
        "french": "Aberrant",
    },
    "input_is_outliers": {
        "english": "Outlier Detected",
        "french": "Valeur Aberrante Détectée",
    },
    "input_values": {
        "english": "Observed",
        "french": "Observations",
    },
    "accepted_values": {
        "english": "Accepted",
        "french": "Filtré",
    },
    "predicted_values": {
        "english": "Predicted",
        "french": "Prédit",
    },
    "predicted_upper_limits": {
        "english": "Upper Bound",
        "french": "Limite Supérieure",
    },
    "predicted_lower_limits": {
        "english": "Lower Bound",
        "french": "Limite inférieure",
    },
    "smoothed": {
        "english": "Smoothed",
        "french": "Lissé",
    },
}


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

    return {x: CLEAN_NAMES[x][language] for x in CLEAN_NAMES}


@dataclass
class UnivariatePlotter:
    signal_name: str
    df: pd.DataFrame
    template: Literal["presentation", "plotly_white"] = field(default="presentation")
    language: Literal["french", "english"] = field(default="english")

    def __post_init__(self):
        df = self.df
        if not df.empty:
            df = align_results_in_time(self.df)
            df = apply_observations_to_outliers(df)
        self.plot_data = df
        self.x = self.plot_data["date"]
        self.names = get_clean_column_names(self.language)

    def lower_limit(self) -> Optional[go.Trace]:
        if "predicted_lower_limits" not in self.plot_data.columns:
            return None
        return go.Scatter(
            x=self.x,
            y=self.plot_data["predicted_lower_limits"],
            name=self.names["predicted_lower_limits"],
            line=dict(width=0),
            mode="lines",
            showlegend=False,
        )

    def upper_limit(self) -> Optional[go.Trace]:
        if "predicted_upper_limits" not in self.plot_data.columns:
            return None
        return go.Scatter(
            x=self.x,
            y=self.plot_data["predicted_upper_limits"],
            name=self.names["predicted_upper_limits"],
            line=dict(width=0),
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(205, 229, 203, 0.5)",
            showlegend=True,
        )

    def input_values(self) -> Optional[go.Trace]:
        if "input_values" not in self.plot_data.columns:
            return None
        return go.Scatter(
            x=self.x,
            y=self.plot_data["input_values"],
            name=self.names["input_values"],
            line=dict(color="black"),
            mode="lines+markers",
            showlegend=True,
        )

    def predicted_values(self) -> Optional[go.Trace]:
        if "predicted_values" not in self.plot_data.columns:
            return None
        return go.Scatter(
            x=self.x,
            y=self.plot_data["predicted_values"],
            name=self.names["predicted_values"],
            line=dict(color="green", dash="dot"),
            mode="lines+markers",
            showlegend=True,
        )

    def outlier_values(self) -> Optional[go.Trace]:
        if "outlier_values" not in self.plot_data.columns:
            return None
        return go.Scatter(
            x=self.x,
            y=self.plot_data["outlier_values"],
            name=self.names["outlier_values"],
            line=dict(color="black"),
            mode="markers",
            marker=dict(symbol="x", size=15),
            showlegend=True,
        )

    def accepted_values(self) -> Optional[go.Trace]:
        if "accepted_values" not in self.plot_data.columns:
            return None
        return go.Scatter(
            x=self.x,
            y=self.plot_data["accepted_values"],
            name=self.names["accepted_values"],
            line=dict(color="#388E3C", dash="dash"),
            mode="lines",
            showlegend=True,
        )

    def smoothed(self) -> Optional[go.Trace]:
        if "smoothed" not in self.plot_data.columns:
            return None
        return go.Scatter(
            x=self.x,
            y=self.plot_data["smoothed"],
            name=self.names["smoothed"],
            line=dict(color="#60FF8F"),
            mode="lines",
            showlegend=True,
        )

    def plot(
        self, series_names: Optional[List[str]] = None, title: Optional[str] = None
    ) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(get_default_plot_elements(self.template))
        if series_names is None:
            series_names = list(self.df.columns)
        self.plot_data = self.df[series_names]

        traces = [
            self.lower_limit(),
            self.upper_limit(),
            self.input_values(),
            self.predicted_values(),
            self.outlier_values(),
            self.accepted_values(),
            self.smoothed(),
        ]
        traces = [trace for trace in traces if trace]
        for trace in traces:
            fig.add_trace(trace)
        if title is None:
            title_start = (
                "Filtration results"
                if self.language == "english"
                else "Résultats de filtration"
            )
            title = f"{title_start}: {self.signal_name}"
        fig.update_layout(
            title=title,
            xaxis=dict(title="Date"),
            yaxis=dict(title="Value" if self.language == "english" else "Valeur"),
        )
        return fig
