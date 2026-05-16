import threading
import webbrowser

import yfinance as yf
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

ticker = "MU"

settings = {
    "1D": ("1d", "1m"),
    "1W": ("5d", "5m"),
    "1M": ("1mo", "1h"),
    "6M": ("6mo", "1d"),
    "1Y": ("1y", "1d"),
    "5Y": ("5y", "1d"),
    "MAX": ("max", "1d"),
}

base_button_style = {
    "marginRight": "6px",
    "backgroundColor": "#222222",
    "color": "white",
    "border": "1px solid #444444",
    "borderRadius": "5px",
    "padding": "6px 12px",
    "cursor": "pointer",
}

active_button_style = {
    **base_button_style,
    "backgroundColor": "#555555",
    "border": "1px solid #aaaaaa",
}


def get_data(range_label):
    period, interval = settings[range_label]

    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        prepost=False,
    )

    if hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)

    return data.dropna()


def make_chart(range_label):
    data = get_data(range_label).copy()
    data["Index"] = range(len(data))

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=data["Index"],
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            customdata=data.index,
            hovertemplate=
                "Date: %{customdata}<br>"
                "Open: %{open}<br>"
                "High: %{high}<br>"
                "Low: %{low}<br>"
                "Close: %{close}<extra></extra>",
            increasing_line_color="#00ff99",
            decreasing_line_color="#ff3333",
            name=ticker,
        )
    )

    tick_count = 6
    tick_positions = data["Index"].iloc[
        [int(i) for i in
         list(__import__("numpy").linspace(0, len(data) - 1, tick_count))]
    ]

    tick_labels = [
        data.index[int(i)].strftime("%b %d")
        for i in tick_positions
    ]

    fig.update_layout(
        title=None,
        template="plotly_dark",
        height=750,
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        font=dict(color="white"),
        xaxis=dict(
            fixedrange=True,
            rangeslider=dict(visible=False),
            tickmode="array",
            tickvals=tick_positions,
            ticktext=tick_labels,
        ),
        yaxis=dict(
            title="Price ($)",
            fixedrange=True,
        ),
        dragmode=False,
    )

    return fig

app = Dash(__name__)

app.layout = html.Div(
    style={
        "backgroundColor": "#111111",
        "padding": "20px",
        "minHeight": "100vh",
    },
    children=[
        html.Div(
            children=[
                html.Span(
                    ticker,
                    style={
                        "color": "white",
                        "fontSize": "22px",
                        "marginRight": "25px",
                    },
                ),
                *[
                    html.Button(label, id=f"btn-{label}", n_clicks=0)
                    for label in settings.keys()
                ],
            ],
            style={
                "marginBottom": "10px",
                "display": "flex",
                "alignItems": "center",
            },
        ),

        dcc.Store(id="selected-range", data="1D"),

        dcc.Graph(
            id="stock-chart",
            figure=make_chart("1D"),
            config={
                "displayModeBar": True,
                "scrollZoom": False,
                "doubleClick": False,
            },
        ),
    ],
)


@app.callback(
    Output("selected-range", "data"),
    [Input(f"btn-{label}", "n_clicks") for label in settings.keys()],
    prevent_initial_call=True,
)
def update_selected_range(*_):
    from dash import ctx
    return ctx.triggered_id.replace("btn-", "")


@app.callback(
    Output("stock-chart", "figure"),
    [Output(f"btn-{label}", "style") for label in settings.keys()],
    Input("selected-range", "data"),
)
def update_chart(range_label):
    styles = [
        active_button_style if label == range_label else base_button_style
        for label in settings.keys()
    ]

    return make_chart(range_label), *styles


def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")


if __name__ == "__main__":
    threading.Timer(1, open_browser).start()
    app.run(debug=False, port=8050)