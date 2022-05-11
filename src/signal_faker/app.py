import json

from dash import Dash, Input, Output, dcc, html

app = Dash(__name__)

app.layout = html.Div(
    [
        dcc.Graph(id="graph"),
        dcc.Interval(
            id="intervals", interval=5 * 1000, n_intervals=0  # in milliseconds
        ),
        html.Br(),
    ]
)


@app.callback(
    Output(component_id="graph", component_property="figure"),
    Input(component_id="intervals", component_property="n_intervals"),
)
def update_fig(_):
    with open("fig.json", "r") as f:
        return json.loads(f.read())


if __name__ == "__main__":
    app.run_server(debug=True)
