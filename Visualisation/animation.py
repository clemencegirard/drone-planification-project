import plotly.graph_objs as go
import numpy as np
from pathlib import Path
import logging

def launch_visualisation_plotly(planning_drones, warehouse):
    drones = list(planning_drones.keys())

    # Palette de couleurs pour les drones
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    # Création de la figure
    fig = go.Figure()

    # Ajouter les éléments
    elements = {
        1: ("grey", "Shelf"),
        2: ("#d1ada6", "Object"),
        3: ("#d1ada6", "Storage line"),
        5: ("yellow", "Starting Mat"),
        6: ("green", "Finish Mat"),
        7: ("brown", "Charging Station")
    }

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

    # Ajouter les trajectoires complètes des drones
    for i, drone_id in enumerate(drones):
        positions = planning_drones[drone_id]['position']
        x, y, z = zip(*positions)
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(width=2, color=color),
            name=f"Trajectoire {drone_id}"
        ))

        fig.add_trace(go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode="markers",
            marker=dict(size=8, color=color),
            name=f"{drone_id}"
        ))

    # Déterminer le nombre de frames maximum
    max_frames = max(len(planning_drones[d]['position']) for d in drones)

    # Créer les frames de l'animation
    frames = []
    slider_steps = []

    for k in range(max_frames):
        frame_data = []

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
                    name = f"path_{drone_id}"
                ))
                frame_data.append(go.Scatter3d(
                    x=[x[k]], y=[y[k]], z=[z[k]],
                    mode="markers",
                    marker=dict(size=4, color=color),
                    name=f"{drone_id}"
                ))

        for elem_id, (color, name) in elements.items():
            elem_positions = np.argwhere(warehouse.mat == elem_id)
            if len(elem_positions) > 0:
                x_elem, y_elem, z_elem = zip(*elem_positions)
                frame_data.append(go.Scatter3d(
                    x=x_elem, y=y_elem, z=z_elem,
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=color,
                        opacity=1,
                        line=dict(  # Contour autour du marker
                            color='DarkSlateGrey',  # Couleur du contour
                            width=2  # Épaisseur du contour
                        )
                    ),
                    name=name
                ))

        frames.append(go.Frame(
            data=frame_data,
            name=str(k),
            layout=go.Layout(title=f"Animation des drones - Temps: {times[min(k, len(times)-1)]}s")
        ))

        slider_steps.append({
            "args": [[str(k)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": str(k),
            "method": "animate"
        })

    fig.frames = frames

    # Ajout du slider pour défiler frame par frame
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

    # Ajout des contrôles d'animation
    fig.update_layout(
        title="Animation des drones",
        scene=dict(
            xaxis=dict(autorange=True),
            yaxis=dict(autorange=True),
            zaxis=dict(autorange=True)
        ),
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True}],
                    "label": "▶️ Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    "label": "⏹ Stop",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "type": "buttons",
            "x": 0.1,
            "y": 0
        }],
        sliders=sliders,  # Ajout du slider
        width = 1200,  # Largeur de la figure
        height = 600,  # Hauteur de la figure
    )

    current_path = Path(__file__).parent.resolve()
    hmtl_dir = current_path / "HTML_files\\drones_animation.html"

    try:
        fig.write_html(hmtl_dir)
        logging.info(f"HTML visualization saved to {hmtl_dir}")

    except Exception as e:
        raise e
