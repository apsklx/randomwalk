import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc


def plot_walks_animated(walks, output_file="random_walk.html", title="Random Walk",
                         theme="plotly_dark", frame_step=1, frame_duration=30, step_length=1.0):
    """
    walks: list of arrays, each shaped (n_steps, dimension), same dimension across all.
    frame_step: draw every Nth step per animation frame (higher = faster/choppier animation).
    frame_duration: ms between frames (lower = faster playback).
    step_length: L, the per-step length used in the expected-distance curve L*sqrt(N).
    """
    dimension = walks[0].shape[1]
    if dimension not in (1, 2, 3):
        raise ValueError("Animation supports 1d/2d/3d only")

    n_steps = walks[0].shape[0]
    colors = pc.sample_colorscale("Turbo", [i / max(len(walks) - 1, 1) for i in range(len(walks))])

    fig = go.Figure()

    # --- walker traces (line + current-position marker per walker) ---
    for i, color in enumerate(colors):
        line_kwargs = dict(mode="lines", name=f"Walker {i+1}",
                            line=dict(width=2 if dimension in (1, 2) else 4, color=color))
        marker_kwargs = dict(mode="markers", marker=dict(size=8, color=color, line=dict(width=1, color="white")),
                              showlegend=False, hoverinfo="skip")
        if dimension in (1, 2):
            fig.add_trace(go.Scatter(x=[], y=[], **line_kwargs))
            fig.add_trace(go.Scatter(x=[], y=[], **marker_kwargs))
        else:
            fig.add_trace(go.Scatter3d(x=[], y=[], z=[], **line_kwargs))
            fig.add_trace(go.Scatter3d(x=[], y=[], z=[], **marker_kwargs))

    n_walker_traces = len(fig.data)  # everything after this index is the expected-distance reference

    # --- expected-distance reference trace(s), dashed ---
    expected_style = dict(line=dict(dash="dash", width=2, color="rgba(255,255,255,0.6)"),
                           name=f"Expected distance ({step_length}·√N)")
    if dimension == 1:
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", **expected_style))
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False,
                                  line=expected_style["line"]))
    elif dimension == 2:
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", **expected_style))
    else:
        theta = np.linspace(0, 2 * np.pi, 60)
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines", **expected_style))
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines", showlegend=False, line=expected_style["line"]))
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines", showlegend=False, line=expected_style["line"]))

    # --- build frames: each frame shows all walks truncated to step k, plus expected-distance shape at k ---
    frame_indices = list(range(2, n_steps + 1, frame_step))
    if frame_indices[-1] != n_steps:
        frame_indices.append(n_steps)

    theta = np.linspace(0, 2 * np.pi, 60)  # reused for 2D/3D circles

    frames = []
    for k in frame_indices:
        frame_data = []

        for walk in walks:
            seg = walk[:k]
            if dimension == 1:
                steps = np.arange(k)
                frame_data.append(go.Scatter(x=steps, y=seg[:, 0]))
                frame_data.append(go.Scatter(x=[steps[-1]], y=[seg[-1, 0]]))
            elif dimension == 2:
                frame_data.append(go.Scatter(x=seg[:, 0], y=seg[:, 1]))
                frame_data.append(go.Scatter(x=[seg[-1, 0]], y=[seg[-1, 1]]))
            else:
                frame_data.append(go.Scatter3d(x=seg[:, 0], y=seg[:, 1], z=seg[:, 2]))
                frame_data.append(go.Scatter3d(x=[seg[-1, 0]], y=[seg[-1, 1]], z=[seg[-1, 2]]))

        r = step_length * np.sqrt(k)  # expected distance at this step count

        if dimension == 1:
            xs = np.arange(k)
            frame_data.append(go.Scatter(x=xs, y=step_length * np.sqrt(xs)))
            frame_data.append(go.Scatter(x=xs, y=-step_length * np.sqrt(xs)))
        elif dimension == 2:
            frame_data.append(go.Scatter(x=r * np.cos(theta), y=r * np.sin(theta)))
        else:
            frame_data.append(go.Scatter3d(x=r * np.cos(theta), y=r * np.sin(theta), z=np.zeros_like(theta)))
            frame_data.append(go.Scatter3d(x=r * np.cos(theta), y=np.zeros_like(theta), z=r * np.sin(theta)))
            frame_data.append(go.Scatter3d(x=np.zeros_like(theta), y=r * np.cos(theta), z=r * np.sin(theta)))

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

    max_r = step_length * np.sqrt(n_steps)  # final expected distance, used to size fixed axes

    if dimension == 1:
        fig.update_layout(xaxis_title="Step", yaxis_title="Position")
        all_y = np.concatenate([w[:, 0] for w in walks] + [np.array([max_r, -max_r])])
        pad_y = 0.05 * (all_y.max() - all_y.min() + 1)
        fig.update_xaxes(range=[0, n_steps])
        fig.update_yaxes(range=[all_y.min() - pad_y, all_y.max() + pad_y])
    elif dimension == 2:
        fig.update_layout(xaxis_title="X", yaxis_title="Y",
                           yaxis=dict(scaleanchor="x", scaleratio=1))
        all_x = np.concatenate([w[:, 0] for w in walks] + [np.array([max_r, -max_r])])
        all_y = np.concatenate([w[:, 1] for w in walks] + [np.array([max_r, -max_r])])
        pad_x, pad_y = 0.05 * (all_x.max() - all_x.min() + 1), 0.05 * (all_y.max() - all_y.min() + 1)
        fig.update_xaxes(range=[all_x.min() - pad_x, all_x.max() + pad_x])
        fig.update_yaxes(range=[all_y.min() - pad_y, all_y.max() + pad_y])
    else:
        fig.update_layout(scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="cube",
            xaxis=dict(range=[-max_r * 1.1, max_r * 1.1]),
            yaxis=dict(range=[-max_r * 1.1, max_r * 1.1]),
            zaxis=dict(range=[-max_r * 1.1, max_r * 1.1]),
        ))

    fig.write_html(output_file, include_plotlyjs="cdn", auto_play=False)
    print(f"Saved to {output_file}")
    return fig


if __name__ == "__main__":
    step_length = 1.5
    from engine import gen_walk

    n_walkers, n_steps, dim = 10, 400, 2
    walks = [gen_walk(n_steps, dim, step_choices=(-step_length, step_length)) for _ in range(n_walkers)]

    plot_walks_animated(
        walks,
        output_file=f"walk_{dim}d.html",
        title=f"{n_walkers} Walkers in {dim}D",
        frame_step=2,
        frame_duration=30,
        step_length = step_length,
    )