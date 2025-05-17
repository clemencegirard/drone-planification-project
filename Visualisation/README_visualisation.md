# Visualization

This module is dedicated to the visualization and animation of drone trajectories within a 3D warehouse environment. It provides interactive tools to analyze and optimize drone movements in real-time.

## 3D Drone Visualization System

`animation.py` is the primary module for generating 3D animated representations of drone trajectories inside a warehouse. Using Plotly, it creates an interactive visualization that helps analyze the navigation and behavior of drones in different scenarios.

### Core Features

#### **Initialization and Configuration**

- **`launch_visualisation_plotly(planning_drones, warehouse)`**  
  Generates a 3D animated visualization of drone movements within a structured warehouse.
  
  - **Parameters:**
    - `planning_drones`: Dictionary containing drone IDs, positions, and timestamps.
    - `warehouse`: Warehouse object containing a 3D matrix representation of the space.

#### **Warehouse Elements Representation**

- The warehouse structure is visualized with distinct colors to represent different elements:
  - **Grey**: Shelves
  - **Brown**: Charging Stations
  - **Yellow**: Start Mat
  - **Green**: Finish Mat
  - **Light Brown**: Objects and Storage Lines

- The function identifies warehouse elements by scanning the matrix and maps them to corresponding colors in the visualization.

#### **Drone Trajectory Visualization**

- Each drone trajectory is plotted using a unique color from a predefined palette.
- The starting position of each drone is highlighted with a marker.
- The entire flight path is represented as a continuous line to track movement.

#### **Animation and Playback Controls**

- **Dynamic Animation Frames**:
  - Frames are generated based on the maximum recorded timestamps of the drones.
  - Each frame updates the drone's position progressively over time.
  
- **Slider and Playback Buttons**:
  - A slider allows manual frame navigation.
  - Play and stop buttons control the animation sequence interactively.
  
#### **Exporting the Visualization**

- **`fig.write_html("HTML_files/drones_animation.html")`**  
  Saves the animated visualization as an interactive HTML file, allowing external review and analysis.

