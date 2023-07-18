from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from data_filters.utilities import apply_observations_to_outliers

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
    "slope": {
        "english": "Slope",
        "french": "Pente",
    },
    "slope_test": {
        "english": "Slope",
        "french": "Pente",
    },
    "failed_slope_test": {
        "english": "Failed Slope Test",
        "french": "Test de Pente Échoué",
    },
    "residuals": {
        "english": "Residuals",
        "french": "Résidus",
    },
    "residuals_test": {
        "english": "Residuals",
        "french": "Résidus",
    },
    "range_test": {
        "english": "Range",
        "french": "Plage",
    },
    "failed_residuals_test": {
        "english": "Failed Residuals Test",
        "french": "Test de Résidus Échoué",
    },
    "correlation_score": {
        "english": "Residual Correlation Score",
        "french": "Score de Corrélation des Résiduels",
    },
    "correlation_test": {
        "english": "Residuals Correlation Score",
        "french": "Score de Corrélation des Résiduels",
    },
    "failed_correlation_test": {
        "english": "Failed Signs Correlation Test",
        "french": "Test de Corrélation des Signes Échoué",
    },
    "failed_range_test": {
        "english": "Failed Range Test",
        "french": "Test de Plage Échoué",
    },
    "accepted": {
        "english": "Accepted",
        "french": "Filtré",
    },
    "rejected": {
        "english": "Rejected",
        "french": "Rejeté",
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
    series_name: str = "series",
    plot_title: str = "title",
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
    fig.update_layout(title=plot_title)
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
        if not self.df.empty:
            self.df = apply_observations_to_outliers(self.df)
        self.plot_data = self.df
        self.x = self.plot_data.index
        self.names = get_clean_column_names(self.language)

    def lower_limit(self) -> Optional[go.Scatter]:
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

    def upper_limit(self) -> Optional[go.Scatter]:
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

    def input_values(self) -> Optional[go.Scatter]:
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

    def predicted_values(self) -> Optional[go.Scatter]:
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

    def outlier_values(self) -> Optional[go.Scatter]:
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

    def accepted_values(self) -> Optional[go.Scatter]:
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

    def smoothed(self) -> Optional[go.Scatter]:
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

    def plot_outlier_results(
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
            xaxis=dict(title="Index"),
            yaxis=dict(title="Value" if self.language == "english" else "Valeur"),
        )
        return fig

    def plot_quality_test_result(
        self,
        test_name: Literal[
            "slope_test", "range_test", "correlation_test", "residuals_test"
        ],
        min_value,
        max_value,
        title: Optional[str] = None,
    ) -> go.Figure:
        col_lookup = {
            "slope_test": {
                "value": "slope",
                "failed_test": "failed_slope_test",
            },
            "range_test": {
                "value": "smoothed",
                "failed_test": "failed_range_test",
            },
            "correlation_test": {
                "value": "correlation_score",
                "failed_test": "failed_correlation_test",
            },
            "residuals_test": {
                "value": "residuals",
                "failed_test": "failed_residuals_test",
            },
        }

        if test_name not in col_lookup.keys():
            raise ValueError(
                f"test_name must be one of {list(col_lookup)} but got {test_name}"
            )

        # Create subplot with 2 y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.update_layout(get_default_plot_elements(self.template))

        df = self.df
        # if the test is the range test, don't plot the accepted values here
        if test_name in ["correlation_test", "slope_test"]:
            # Line for smoothed data
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["smoothed"],
                    mode="lines",
                    name=CLEAN_NAMES["smoothed"][self.language],
                ),
                secondary_y=False,
            )
        if test_name == ["correlation_test"]:
            # Line for smoothed data
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["accepted_values"],
                    mode="lines",
                    name=CLEAN_NAMES["accepted_values"][self.language],
                ),
                secondary_y=False,
            )

        # if the test is the signs test or the slope test, we plot the absolute values
        if test_name in ["correlation_test", "slope_test", "residuals_test"]:
            df[col_lookup[test_name]["value"]] = np.abs(
                df[col_lookup[test_name]["value"]]
            )
        # Scatter plot for slope with color depending on 'failed_slope_test'
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col_lookup[test_name]["value"]],
                mode="markers",
                name=CLEAN_NAMES[test_name][self.language],
                marker=dict(
                    color=df[col_lookup[test_name]["failed_test"]].map(
                        {True: "red", False: "green"}
                    ),  # Map 'failed_slope_test' to colors
                    size=7,
                ),
            ),
            secondary_y=test_name in ["slope_test", "correlation_test"],
        )

        # Add constant lines for min_slope and max_slope
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[min_value] * np.ones(len(df.index)),
                mode="lines",
                name="Min",
                marker=dict(
                    color="red",
                ),
            ),
            secondary_y=test_name in ["slope_test", "correlation_test"],
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=max_value * np.ones(len(df.index)),
                mode="lines",
                name="Max",
                marker=dict(
                    color="red",
                    # make it dashed
                ),
            ),
            secondary_y=test_name in ["slope_test", "correlation_test"],
        )
        # Set titles
        fig.update_layout(
            title_text=f"{test_name.title()} results for {self.signal_name}"
        )

        fig.update_yaxes(title_text="Accepted Values", secondary_y=False)

        if test_name in ["slope_test", "correlation_test"]:
            fig.update_yaxes(
                title_text=CLEAN_NAMES[col_lookup[test_name]["value"]][self.language],
                secondary_y=True,
            )

        return fig

    def plot_original_and_final_data(
        self,
    ) -> go.Figure:
        df = self.df

        fig = go.Figure()
        fig.update_layout(get_default_plot_elements(self.template))
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["input_values"],
                mode="lines",
                name=CLEAN_NAMES["input_values"][self.language],
                marker=dict(
                    color="blue",
                ),
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["accepted"],
                mode="markers",
                name=CLEAN_NAMES["accepted"][self.language],
                marker=dict(
                    color="green",
                ),
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["rejected"],
                mode="markers",
                name=CLEAN_NAMES["rejected"][self.language],
                marker=dict(
                    color="red",
                ),
            ),
        )
        fig.update_layout(title_text=f"Original and Final Data for {self.signal_name}")
        return fig


@dataclass
class MultivariatePlotter:
    df: pd.DataFrame
    signal_names: list[str] = field(default_factory=list)
    template: Literal["presentation", "plotly_white"] = field(default="presentation")
    language: Literal["french", "english"] = field(default="english")

    def __post_init__(self):
        self.plot_data = self.df
        self.x = self.plot_data.index
        self.names = get_clean_column_names(self.language)

    def plot_2_main_components(self, title: Optional[str] = None) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(get_default_plot_elements(self.template))
        fig.add_trace(
            go.Scatter(
                x=self.plot_data["PC_1"],
                y=self.plot_data["PC_2"],
                mode="markers",
                marker=dict(
                    color=self.plot_data["is_rejected"].map(
                        {True: "red", False: "green"}
                    ),  # Map 'failed_slope_test' to colors
                    size=7,
                ),
            )
        )

        fig.update_xaxes(title_text="PC_1")
        fig.update_yaxes(title_text="PC_2")
        if title is None:
            fig.update_layout(
                title_text=f"2 Main Components for {', '.join(self.signal_names)}"
            )
        else:
            fig.update_layout(title_text=title)
        return fig

    def plot_test_results(
        self,
        test_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(get_default_plot_elements(self.template))
        fig.add_trace(
            go.Scatter(
                x=self.x,
                y=self.plot_data[test_name],
                mode="lines",
                name=test_name,
                marker=dict(
                    color=self.plot_data[f"failed_{test_name}"].map(
                        {True: "red", False: "green"}
                    ),  # Map 'failed_slope_test' to colors
                    size=7,
                ),
            )
        )
        if min_value is not None:
            fig.add_trace(
                go.Scatter(
                    x=self.x,
                    y=[min_value] * np.ones(len(self.x)),
                    mode="lines",
                    name="Min",
                    marker=dict(
                        color="red",
                    ),
                )
            )
        if max_value is not None:
            fig.add_trace(
                go.Scatter(
                    x=self.x,
                    y=[max_value] * np.ones(len(self.x)),
                    mode="lines",
                    name="Max",
                    marker=dict(
                        color="red",
                    ),
                )
            )
        if title is None:
            fig.update_layout(
                title_text=f"{test_name} results for {', '.join(self.signal_names)}"
            )
        else:
            fig.update_layout(title_text=title)
        return fig
