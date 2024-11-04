import plotly.graph_objects as go
import numpy as np


def create_gauge_chart(probabilities):
    avg_probability = (
        np.mean(list(probabilities.values())) * 100
    )  # Convert to percentage

    if avg_probability < 30:
        color = "rgba(0, 255, 0, 0.3)"
    elif avg_probability < 60:
        color = "rgba(255, 255, 0, 0.3)"
    else:
        color = "rgba(255, 0, 0, 0.3)"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=avg_probability,
            domain={
                "x": [0, 1],
                "y": [0, 1],
            },
            title={
                "text": "Average Probability of Churn",
                "font": {"size": 24, "color": "white"},
            },
            number={
                "font": {
                    "size": 48,
                    "color": "white",
                },
                "suffix": "%",
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "darkblue",
                },
                "bar": {
                    "color": color,
                },
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 2,
                "bordercolor": "white",
                "steps": [
                    {
                        "range": [0, 30],
                        "color": "rgba(0, 255, 0, 0.3)",
                    },
                    {
                        "range": [30, 60],
                        "color": "rgba(255, 255, 0, 0.3)",
                    },
                    {
                        "range": [60, 100],
                        "color": "rgba(255, 0, 0, 0.3)",
                    },
                ],
                "threshold": {
                    "line": {
                        "color": "white",
                        "width": 4,
                    },
                    "thickness": 0.75,
                    "value": 100,
                },
            },
        )
    )

    # Update chart layout
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        width=400,
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def create_probability_chart(probabilities):
    models = list(probabilities.keys())
    probs = [prob * 100 for prob in probabilities.values()]  # Convert to percentages

    fig = go.Figure(
        data=[
            go.Bar(
                y=models,
                x=probs,
                orientation="h",
                text=[f"{p:.2f}%" for p in probs],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="Churn Probability by Model",
        yaxis_title="Models",
        xaxis_title="Probability",
        xaxis=dict(tickformat=".0%", range=[0, 100]),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )

    return fig
