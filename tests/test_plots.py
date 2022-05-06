import pandas as pd
import pytest
from filters.plots import (UnivariatePlotter, add_traces_to_other_plot,
                           plot_array)

from test_filters import get_data


def get_result_df(path: str = "tests/test_filter_results.csv"):
    return pd.read_csv(path, index_col=0)


def test_plot_array():
    try:
        data = get_data("clean sine").to_numpy()
        plot_array(data, "clean sine", "Test plot", template="plotly_white")
        # fig.show()
    except Exception:
        assert False


def test_add_traces_to_other_plots():
    try:
        data1 = get_data("clean sine").to_numpy()
        fig1 = plot_array(data1, "clean sine", "Test plot", template="plotly_white")
        data2 = get_data("dirty sine").to_numpy()
        fig2 = plot_array(data2, "dirty sine", "Test plot2", template="plotly_white")
        add_traces_to_other_plot(fig1, fig2)
        # fig3.show()
    except Exception:
        assert False


@pytest.mark.parametrize("language", [("english"), ("french")])
def test_plot_results(language):
    df = get_result_df()
    plotter = UnivariatePlotter(
        signal_name="dirty sine + shift", df=df, language=language
    )
    plotter.plot(title="dirty sine + shift")  #.show()
