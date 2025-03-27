# Planification generation

This module is dedicated to the creation of the planning for each drone. It includes scripts for calculating the adjacency matrix in case of Bellman algorithm usage, task list generation and more.

## Drone Mission Planning System

`planification.py` is the core module that handles comprehensive drone mission planning in warehouse environments, including task management, route optimization, and battery management.

### Core Functions

#### Configuration and Initialization

- **`load_config_planning(planning_name, config_path)`**  
  Loads planning configuration from JSON file for a specific planning scenario.

- **`open_csv(csv_file_name)`**  
  Reads and parses the task list CSV file.

- **`prepare_csv(csv_file_name, warehouse, config)`**  
  Prepares task data by computing distances and travel times.

#### Task Processing

- **`full_task_treatment()`**  
  Handles complete task processing including positioning, execution, and return-to-charge.

- **`task_processing()`**  
  Manages task execution and updates drone schedule.

- **`positionning()`**  
  Calculates optimal positioning path for drones before task execution.

#### Path Management

- **`add_path_to_df()`**  
  Adds computed paths to drone schedule DataFrame.

- **`compute_travel_time()`**  
  Calculates travel time between two points using Manhattan distance.

#### Drone Selection and Scheduling

- **`drone_initial_selection()`**  
  Selects optimal drone for a task based on battery and availability.

- **`drone_empty_battery_selection()`**  
  Handles task assignment when drones require charging.

- **`create_drone_schedule()`**  
  Main function that creates complete schedule for all drones.

#### Battery Management

- **`delta_battery()`**  
  Calculates required charging time and battery gain.

- **`drone_charging_process()`**  
  Manages the charging process for drones.



## Task List Generator

`task_list_generator.py` generates and manages task lists for drone operations in a warehouse environment. It handles both inbound (arrival) and outbound (departure) logistics operations.

### Core Functions

- **`generate_object_id(length=8) -> str`**  
  Generates a random alphanumeric ID for warehouse objects using uppercase letters and digits.

- **`create_objects_in_warehouse(n_objects: int, warehouse: Warehouse3D) -> list[Object]`**  
  Creates and places objects in the warehouse with random distribution between shelves and arrival slots:
  - Randomly assigns objects to available shelf positions (warehouse.mat value = 2)
  - Places remaining objects on arrival slots (warehouse.mat value = 5)
  - Returns list of created `Object` instances

- **`choose_slot_and_time(times_positions: dict[time, dict[tuple[int,int,int], int]]) -> tuple`**  
  Helper function that selects an available time slot and position based on capacity:
  - Processes time slots in chronological order
  - Returns first available (time, position) tuple with remaining capacity
  - Returns (None, None) if no slots are available

- **`generate_task_list(n_tasks: int, objects: list[Object], arrival_times: list[time], departure_times: list[time], warehouse: Warehouse3D, mapping_config) -> str`**  
  Main function that generates a CSV task list with the following columns:




## Bellman Algorithm Implementation

`bellman.py` implements the Bellman-Ford algorithm for finding shortest paths in a weighted graph, specifically designed for drone path planning in warehouse environments.

### Core Functions

- **`inf_mat(M: np.ndarray) -> np.ndarray`**  
  Prepares the adjacency matrix for Bellman-Ford processing by:
  - Converting all zero values (except diagonal) to infinity
  - Preserving existing non-zero values representing valid edges

- **`bellman_recursif(noeud_depart: int, M: list, dist: list, pred: list, iterations: int) -> list`**  
  Recursive implementation of Bellman-Ford algorithm:
  - Processes nodes in order starting with departure node
  - Updates distance and predecessor lists when shorter paths are found
  - Terminates early if no improvements are made (optimization)
  - Returns final distance list for the given starting node

- **`minimum_distance_matrix(M) -> np.ndarray`**  
  Computes all-pairs shortest paths using:
  - Progress tracking with `tqdm` for large matrices
  - Calls `bellman_recursif` for each node as starting point
  - Returns complete distance matrix (n x n) where C[i,j] = shortest distance from i to j

- **`main_bellman(M) -> np.ndarray`**  
  Main wrapper function that:
  1. Prepares matrix with `inf_mat`
  2. Computes all shortest paths with `minimum_distance_matrix`
  3. Returns complete distance matrix


## Warehouse Adjacency Matrix Generator

`adjacency_matrix.py` creates and manages adjacency matrices for warehouse path planning using Manhattan distance calculations with BFS (Breadth-First Search).

### Core Functions

#### Matrix Generation Functions

- **`extract_coordinates(matrix, values) -> dict`**  
  Extracts coordinates from 3D warehouse matrix for specified values:
  - Returns dictionary mapping values to coordinate lists
  - Handles multiple value types (0-4) representing different warehouse zones

- **`generate_adjacency_matrix(warehouse_3d, coordinates) -> np.ndarray`**  
  Generates full adjacency matrix for a coordinate set:
  - Uses Manhattan distance with BFS pathfinding
  - Creates symmetric matrix (n x n) where n = coordinate count
  - Diagonal contains zeros (self-distance)

- **`generate_diagonal_checkpoints_adjmatrix(warehouse_3d, coordinates, block_size=25)`**  
  Optimized matrix generation for checkpoints:
  - Processes in blocks to improve performance
  - Uses precomputed checkpoint connections
  - Includes progress tracking with `tqdm`

#### Matrix Assembly Functions

- **`update_with_inter_category_distances()`**  
  Computes distances between different warehouse categories:
  - Handles specific category pairs (object/checkpoint, storage_line/checkpoint, etc.)
  - Updates global matrix with symmetric values

- **`assemble_global_adjacency_matrix()`**  
  Combines individual category matrices into global matrix:
  1. Places diagonal blocks for each category
  2. Fills inter-category distances
  3. Returns complete (N x N) matrix where N = total locations

#### Main Functions

- **`main_adjacency(warehouse_3d, category_mapping)`**  
  Main workflow controller:
  1. Extracts coordinates by category
  2. Generates individual matrices
  3. Assembles global matrix
  4. Returns:
     - Complete adjacency matrix
     - Coordinate-to-index mapping dictionary

- **`save_adj_matrix(adjacency_matrix, warehouse_name)`**  
  Saves matrix to CSV in AMatrix directory:
  - Creates directory if missing
  - Uses naming convention: `AM_{warehouse_name}.csv`
