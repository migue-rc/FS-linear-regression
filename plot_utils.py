import matplotlib.pyplot as plt

assets_folder = './assets'

def plot_fit(X, y_true, predicted, title, show_error=False):
    plt.scatter(X, y_true, label='Actual')
    plt.plot(X, predicted, color='orange', label='Predicted')

    if show_error:
        plt.vlines(X, y_true, predicted, colors='red', linestyles='dashed', alpha=0.5, label='Error')

    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_loss(loss_history):
    plt.plot(loss_history)
    plt.title("Loss History")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_slope(m, b):
    X = [1, 10]
    y = [m + b, (10 * m) + b]
    plt.grid(True)
    plt.plot(X, y)
    plt.show()


import plotly.io as pio
pio.renderers.default = "notebook"
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def create_interactive_line_chart(dataset, scatter_x=None, scatter_y=None, x_range=(-10, 10)):
    """
    Creates an interactive Plotly chart with a slider.
    Optionally plots static points provided via scatter_x and scatter_y.
    """
    # Standardize input to a DataFrame
    if not isinstance(dataset, pd.DataFrame):
        df = pd.DataFrame(dataset, columns=['m', 'b'])
    else:
        df = dataset.copy()

    # Generate X values
    x_vals = np.linspace(x_range[0], x_range[1], 100)

    # Calculate global y-axis limits using the standard y = mx + b formula
    y_at_x_start = df['m'] * x_range[0] + df['b']
    y_at_x_end = df['m'] * x_range[1] + df['b']

    y_min = min(y_at_x_start.min(), y_at_x_end.min())
    y_max = max(y_at_x_start.max(), y_at_x_end.max())

    # Include static points in the y-axis calculation so they fit in the view
    if scatter_y is not None:
        y_min = min(y_min, min(scatter_y))
        y_max = max(y_max, max(scatter_y))

    y_padding = (y_max - y_min) * 0.05
    # Fallback padding just in case the line is perfectly flat
    if y_padding == 0:
        y_padding = 1

    y_axis_range = [y_min - y_padding, y_max + y_padding]

    # Initialize the figure
    fig = go.Figure()

    # Add the initial line trace (Trace 0)
    initial_m = df['m'].iloc[0]
    initial_b = df['b'].iloc[0]
    initial_y = initial_m * x_vals + initial_b
    initial_name = f"m = {initial_m:.2f}, b = {initial_b:.2f}"

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=initial_y,
        mode='lines',
        name=initial_name,
        line=dict(color='blue', width=3)
    ))

    # Add the static points trace (Trace 1)
    if scatter_x is not None and scatter_y is not None:
        fig.add_trace(go.Scatter(
            x=scatter_x,
            y=scatter_y,
            mode='markers',
            name='Actual Data',
            marker=dict(color='red', size=10, symbol='circle')
        ))

    # Build the steps for the slider
    steps = []
    for i, row in df.iterrows():
        m = row['m']
        b = row['b']

        # Pure y = mx + b
        y_vals = m * x_vals + b
        line_eq = f"m = {m:.2f}, b = {b:.2f}"

        step = dict(
            method="restyle",
            args=[
                {"y": [y_vals], "name": [line_eq]},
                [0]  # CRITICAL: Tells Plotly to only update Trace 0 (the line)
            ],
            label=str(i)
        )
        steps.append(step)

    # Configure the slider
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Iteration: "},
        pad={"t": 50},
        steps=steps
    )]

    # Update layout with slider and fixed axes
    fig.update_layout(
        sliders=sliders,
        title="Interactive Linear Regression Fit",
        xaxis_title="X",
        yaxis_title="Y",
        xaxis=dict(range=[x_range[0], x_range[1]], autorange=False),
        yaxis=dict(range=y_axis_range, autorange=False),
        template="plotly_white"
    )
    fig.write_html(f"{assets_folder}/line_chart.html")

    fig.show()


def plot_interactive_loss(loss_history):
    """
    Creates an interactive Plotly chart for the loss history.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(loss_history))),
        y=loss_history,
        mode='lines',
        name='MSE Loss',
        line=dict(color='firebrick', width=2)
    ))

    fig.update_layout(
        title="Loss History Over Epochs",
        xaxis_title="Epoch",
        yaxis_title="Mean Squared Error (MSE)",
        template="plotly_white",
        hovermode="x unified"
    )


    fig.write_html(f"{assets_folder}/loss_history.html")
    fig.show()

