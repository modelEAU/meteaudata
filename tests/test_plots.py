from filters.plots import add_traces_to_other_plot, plot_array

from test_filters import get_data


def test_plot_array():
    try:
        data = get_data("clean sine")
        fig = plot_array(data, "clean sine", "Test plot", mode="plotly_white")
        fig.show()
    except Exception:
        assert False


def test_add_traces_to_other_plots():
    try:
        data1 = get_data("clean sine")
        fig1 = plot_array(data1, "clean sine", "Test plot", mode="plotly_white")
        data2 = get_data("dirty sine")
        fig2 = plot_array(data2, "dirty sine", "Test plot2", mode="plotly_white")
        fig3 = add_traces_to_other_plot(fig1, fig2)
        fig3.show()
    except Exception:
        assert False
