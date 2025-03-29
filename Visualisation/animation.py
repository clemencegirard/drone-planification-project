import plotly.graph_objs as go
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple


def launch_visualisation_plotly(planning_drones: Dict[str, Dict[str, List[Tuple[int, int, int]]]],
                                warehouse: object) -> None:
    """
    Generates a 3D animated visualization of drone trajectories within a warehouse.

    Parameters:
    - planning_drones: Dictionary containing drone IDs as keys and a dictionary of positions and timestamps as values.
    - warehouse: Object containing the warehouse matrix (`warehouse.mat`) representing elements in the space.

    The function creates an interactive Plotly visualization, saving it as an HTML file.
    """
    drones: List[str] = list(planning_drones.keys())

    # Color palette for drones
    colors: List[str] = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    # Create the figure
    fig = go.Figure()

    # Define warehouse elements
    elements: Dict[int, Tuple[str, str]] = {
        1: ("grey", "Shelf"),
        2: ("#d1ada6", "Object"),
        3: ("#d1ada6", "Storage line"),
        5: ("yellow", "Starting Mat"),
        6: ("green", "Finish Mat"),
        7: ("brown", "Charging Station")
    }

    # Add warehouse elements to the plot
    for elem_id, (color, name) in elements.items():
        elem_positions = np.argwhere(warehouse.mat == elem_id)
        if len(elem_positions) > 0:
            x_elem, y_elem, z_elem = zip(*elem_positions)
            fig.add_trace(go.Scatter3d(
                x=x_elem, y=y_elem, z=z_elem,
                mode="markers",
                marker=dict(size=5, color=color, opacity=0.9),
                name=name
            ))

    # Add drone trajectories
    for i, drone_id in enumerate(drones):
        positions = planning_drones[drone_id]['position']
        x, y, z = zip(*positions)
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(width=2, color=color),
            name=f"Trajectory {drone_id}"
        ))

        fig.add_trace(go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode="markers",
            marker=dict(size=8, color=color),
            name=f"{drone_id}"
        ))

    # Determine the maximum number of frames
    max_frames: int = max(len(planning_drones[d]['position']) for d in drones)

    # Create animation frames
    frames: List[go.Frame] = []
    slider_steps: List[Dict] = []

    for k in range(max_frames):
        frame_data: List[go.Scatter3d] = []

        for i, drone_id in enumerate(drones):
            positions = planning_drones[drone_id]['position']
            times = planning_drones[drone_id]['time']
            x, y, z = zip(*positions)
            color = colors[i % len(colors)]

            if k < len(x):
                frame_data.append(go.Scatter3d(
                    x=x[:k + 1], y=y[:k + 1], z=z[:k + 1],
                    mode="lines",
                    line=dict(width=2, color=color),
                    name=f"path_{drone_id}"
                ))
                frame_data.append(go.Scatter3d(
                    x=[x[k]], y=[y[k]], z=[z[k]],
                    mode="markers",
                    marker=dict(size=4, color=color),
                    name=f"{drone_id}"
                ))

        frames.append(go.Frame(
            data=frame_data,
            name=str(k),
            layout=go.Layout(title=f"Drone Animation - Time: {times[min(k, len(times) - 1)]}s")
        ))

        slider_steps.append({
            "args": [[str(k)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": str(k),
            "method": "animate"
        })

    fig.frames = frames

    # Add slider for manual frame navigation
    sliders = [{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "prefix": "Frame: ",
            "font": {"size": 14, "color": "black"}
        },
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": slider_steps
    }]

    # Add animation controls
    fig.update_layout(
        title="Drone Animation",
        scene=dict(
            xaxis=dict(autorange=True),
            yaxis=dict(autorange=True),
            zaxis=dict(autorange=True)
        ),
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True}],
                 "label": "▶️ Play", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                 "label": "⏹ Stop", "method": "animate"}
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "type": "buttons",
            "x": 0.1,
            "y": 0
        }],
        sliders=sliders,
        width=1200,
        height=600,
    )

    # Save the visualization as an HTML file
    current_path = Path(__file__).parent.resolve()
    html_dir = current_path / "HTML_files/drones_animation.html"

    # Ensure the directory exists
    html_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.write_html(str(html_dir))
        logging.info(f"HTML visualization saved to {html_dir}")
    except Exception as e:
        logging.error(f"Failed to save HTML file: {e}")
        raise