import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc


def plot_walks_animated_1d(walks, ci_intervals, rms_values, output_file="random_walk.html",
                            title="Random Walk", theme="plotly_dark",
                            frame_step=1, frame_duration=30, confidence_level=0.90):
    """
    walks: list of arrays, each shaped (n_steps, 1).
    ci_intervals: array of shape (n_steps, 2) from engine.confidence_interval — [:,0]=lower, [:,1]=upper.
    rms_values: array of shape (n_steps,) from engine.rms_distance, one value per step.
    Purely a drawing function — no statistics computed here.
    """
    dimension = walks[0].shape[1]
    if dimension != 1:
        raise ValueError("This function only handles 1d walks")

    n_steps = walks[0].shape[0]
    colors = pc.sample_colorscale("Turbo", [i / max(len(walks) - 1, 1) for i in range(len(walks))])

    fig = go.Figure()

    # --- walker traces (line + current-position marker per walker) ---
    for i, color in enumerate(colors):
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name=f"Walker {i+1}",
                                  line=dict(width=2, color=color)))
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers",
                                  marker=dict(size=8, color=color, line=dict(width=1, color="white")),
                                  showlegend=False, hoverinfo="skip"))

    # --- CI boundary traces, dashed ---
    ci_label = f"{int(confidence_level*100)}% CI"
    ci_line_style = dict(dash="dash", width=2, color="rgba(255,255,255,0.6)")
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name=ci_label, line=ci_line_style))
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, line=ci_line_style))

    # --- RMS trace, dotted (distinct style from CI) ---
    rms_line_style = dict(dash="dot", width=2, color="rgba(255,215,0,0.7)")
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="RMS distance", line=rms_line_style))
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, line=rms_line_style))

    # --- build frames: each frame shows all walks truncated to step k, plus CI/RMS up to k ---
    frame_indices = list(range(2, n_steps + 1, frame_step))
    if frame_indices[-1] != n_steps:
        frame_indices.append(n_steps)

    frames = []
    for k in frame_indices:
        frame_data = []

        for walk in walks:
            seg = walk[:k]
            steps = np.arange(k)
            frame_data.append(go.Scatter(x=steps, y=seg[:, 0]))
            frame_data.append(go.Scatter(x=[steps[-1]], y=[seg[-1, 0]]))

        xs = np.arange(k)
        frame_data.append(go.Scatter(x=xs, y=ci_intervals[:k, 1]))   # upper CI
        frame_data.append(go.Scatter(x=xs, y=ci_intervals[:k, 0]))   # lower CI
        frame_data.append(go.Scatter(x=xs, y=rms_values[:k]))        # +RMS
        frame_data.append(go.Scatter(x=xs, y=-rms_values[:k]))       # -RMS

        frames.append(go.Frame(data=frame_data, name=str(k)))

    fig.frames = frames

    fig.update_layout(
        template=theme,
        title=dict(text=title, x=0.5, font=dict(size=22)),
        height=750,
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title="Step",
        yaxis_title="Position",
        updatemenus=[dict(
            type="buttons", showactive=False, y=1.05, x=0.02, xanchor="left",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=frame_duration, redraw=True), fromcurrent=True)]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            steps=[dict(method="animate", args=[[str(k)], dict(mode="immediate",
                        frame=dict(duration=0, redraw=True))], label=str(k)) for k in frame_indices],
            transition=dict(duration=0), x=0.1, len=0.85,
        )],
    )

    # --- fixed axis ranges, sized to whichever boundary (CI or RMS) reaches furthest ---
    all_y = np.concatenate([w[:, 0] for w in walks] + [ci_intervals[:, 0], ci_intervals[:, 1],
                                                        rms_values, -rms_values])
    pad_y = 0.05 * (all_y.max() - all_y.min() + 1)
    fig.update_xaxes(range=[0, n_steps])
    fig.update_yaxes(range=[all_y.min() - pad_y, all_y.max() + pad_y])

    fig.write_html(output_file, include_plotlyjs="cdn", auto_play=False)
    print(f"Saved to {output_file}")
    return fig

def plot_walks_animated_2d3d(walks, output_file="random_walk.html", title="Random Walk",
                              theme="plotly_dark", frame_step=1, frame_duration=30):
    """
    walks: list of arrays, each shaped (n_steps, dimension), dimension is 2 or 3.
    No statistical overlays — just the walkers themselves.
    """
    dimension = walks[0].shape[1]
    if dimension not in (2, 3):
        raise ValueError("This function only handles 2d/3d walks")

    n_steps = walks[0].shape[0]
    colors = pc.sample_colorscale("Turbo", [i / max(len(walks) - 1, 1) for i in range(len(walks))])
    trace_type = go.Scatter if dimension == 2 else go.Scatter3d

    fig = go.Figure()
    for i, color in enumerate(colors):
        line_kwargs = dict(mode="lines", name=f"Walker {i+1}",
                            line=dict(width=2 if dimension == 2 else 4, color=color))
        marker_kwargs = dict(mode="markers", marker=dict(size=8, color=color, line=dict(width=1, color="white")),
                              showlegend=False, hoverinfo="skip")
        if dimension == 2:
            fig.add_trace(trace_type(x=[], y=[], **line_kwargs))
            fig.add_trace(trace_type(x=[], y=[], **marker_kwargs))
        else:
            fig.add_trace(trace_type(x=[], y=[], z=[], **line_kwargs))
            fig.add_trace(trace_type(x=[], y=[], z=[], **marker_kwargs))

    frame_indices = list(range(2, n_steps + 1, frame_step))
    if frame_indices[-1] != n_steps:
        frame_indices.append(n_steps)

    frames = []
    for k in frame_indices:
        frame_data = []
        for walk in walks:
            seg = walk[:k]
            if dimension == 2:
                frame_data.append(go.Scatter(x=seg[:, 0], y=seg[:, 1]))
                frame_data.append(go.Scatter(x=[seg[-1, 0]], y=[seg[-1, 1]]))
            else:
                frame_data.append(go.Scatter3d(x=seg[:, 0], y=seg[:, 1], z=seg[:, 2]))
                frame_data.append(go.Scatter3d(x=[seg[-1, 0]], y=[seg[-1, 1]], z=[seg[-1, 2]]))
        frames.append(go.Frame(data=frame_data, name=str(k)))

    fig.frames = frames

    fig.update_layout(
        template=theme,
        title=dict(text=title, x=0.5, font=dict(size=22)),
        height=750,
        margin=dict(l=10, r=10, t=60, b=10),
        updatemenus=[dict(
            type="buttons", showactive=False, y=1.05, x=0.02, xanchor="left",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=frame_duration, redraw=True), fromcurrent=True)]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            steps=[dict(method="animate", args=[[str(k)], dict(mode="immediate",
                        frame=dict(duration=0, redraw=True))], label=str(k)) for k in frame_indices],
            transition=dict(duration=0), x=0.1, len=0.85,
        )],
    )

    if dimension == 2:
        fig.update_layout(xaxis_title="X", yaxis_title="Y",
                           yaxis=dict(scaleanchor="x", scaleratio=1))
        all_x = np.concatenate([w[:, 0] for w in walks])
        all_y = np.concatenate([w[:, 1] for w in walks])
        pad_x, pad_y = 0.05 * (all_x.max() - all_x.min() + 1), 0.05 * (all_y.max() - all_y.min() + 1)
        fig.update_xaxes(range=[all_x.min() - pad_x, all_x.max() + pad_x])
        fig.update_yaxes(range=[all_y.min() - pad_y, all_y.max() + pad_y])
    else:
        fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"))

    fig.write_html(output_file, include_plotlyjs="cdn", auto_play=False)
    print(f"Saved to {output_file}")
    return fig

if __name__ == "__main__":
    from engine import gen_walk

    step_length = 2.0
    n_walkers, n_steps, dim = 10, 400, 1  # <- change dim to 1, 2, or 3

    walks = [gen_walk(n_steps, dim, step_choices=(-step_length, step_length)) for _ in range(n_walkers)]

    if dim == 1:
        from engine import confidence_interval, rms_distance
        ci = confidence_interval(n_steps, dim, step_length, 0.90)
        rms = np.array([rms_distance(k, step_length) for k in range(n_steps)])
        plot_walks_animated_1d(walks, ci, rms, output_file=f"walk_{dim}d.html",
                                title=f"{n_walkers} Walkers in {dim}D", frame_step=2, frame_duration=30)
    else:
        plot_walks_animated_2d3d(walks, output_file=f"walk_{dim}d.html",
                                  title=f"{n_walkers} Walkers in {dim}D", frame_step=2, frame_duration=30)