# Avoidance of collisisons and Planning optimisation

This module is dedicated to optimizing drone trajectories while minimizing collisions using simulated annealing. It includes functionalities for detecting and mitigating both direct and indirect collisions, as well as near misses.

## Optimisation

`optimisation.py` implements the simulated annealing algorithm to optimize drone planning, focusing on minimizing collisions and total flight duration. Its functions work together to iteratively improve the scheduling solution.

- **`simulated_annealing(...) -> Dict[str, pd.DataFrame]`**  
  Runs the optimization process by iteratively modifying flight plans. It employs the Metropolis acceptance criterion to decide whether to accept a new solution based on its cost, allowing the algorithm to escape local minima.

- **`find_optimal_solution(...) -> Tuple[Dict[str, pd.DataFrame], float, bool]`**  
  Recursively searches for an optimal trajectory solution that minimizes collision risks and overall flight cost. The search terminates when a valid, collision-free plan is found or when a maximum iteration limit is reached.

- **`metropolis_acceptance(current_cost: float, new_cost: float, temp: float) -> bool`**  
  Implements the acceptance criterion that evaluates whether a new planning solution should be accepted, based on cost differences and the current temperature.

- **`plot_evolution(data: list, xlabel: str, ylabel: str, title: str, y_log=False, save_path=None)`**  
  Generates and saves plots to visualize the evolution of the cost function, temperature decay, and acceptance rate during the optimization process, providing valuable insights into the algorithm's performance.

## Avoidance

`avoidance.py` is responsible for collision detection and avoidance strategies. It includes a range of important functions, not only for identifying potential conflicts but also for actively modifying drone trajectories to achieve a collision-free plan.

- **`get_segments(df: pd.DataFrame) -> list`**  
  Generates trajectory segments from recorded drone positions by extracting sequential waypoints from a DataFrame.

- **`count_direct_collisions(drone_data: Dict[str, pd.DataFrame], charging_station_position: tuple) -> pd.DataFrame`**  
  Identifies direct collisions where drones share the same position at the same time. Collisions occurring at the charging station are excluded.

- **`count_calculated_collisions(drone_data: Dict[str, pd.DataFrame], drone_speed: float, charging_station_position: tuple, time_step: float) -> pd.DataFrame`**  
  Detects potential future collisions by interpolating drone positions along their trajectories and checking for intersections.

- **`detect_near_misses(drone_data, drone_speed, charging_station_position, threshold, time_step) -> pd.DataFrame`**  
  Flags instances where drones come dangerously close (based on a proximity threshold) without actually colliding.

- **`compute_cost(drone_data: Dict[str, pd.DataFrame], drone_speed: int, charging_station_position: tuple, threshold: int, time_step: float, collision_penalty: float, avoidance_penalty: float, total_duration_penalty: float) -> float`**  
  Computes the overall cost of a given flight plan by considering total flight duration, collision penalties, and near-miss penalties.

- **`change_planning(planning: Dict[str, pd.DataFrame], heuristic: int, direct_collisions: pd.DataFrame, calculated_collisions: pd.DataFrame, near_misses: pd.DataFrame)`**  
  Modifies an existing drone planning using heuristics tailored to resolve detected conflicts (direct collisions, calculated collisions, or near misses).

- **Delaying Transit Times**  
  - **`push_back_transit_times(planning: pd.DataFrame, collision_time: pd.Timestamp, time_offset: int)`**  
    This function identifies the time interval around a detected collision and delays all subsequent transit times between two charging station visits. By shifting the schedule, it helps prevent temporal overlaps that could lead to collisions.

- **Bypassing Obstacles**  
  - **`bypass_obstacle(planning: pd.DataFrame, start_pos: tuple[int, int, int], start_pos_time: str, dimension_index: int, drone_speed: int) -> pd.DataFrame`**  
    Adjusts a drone's trajectory to avoid an imminent collision. It creates new intermediate positions—offset by a small, random value—and adjusts the corresponding transit times, effectively rerouting the drone to bypass potential obstacles.

- **Heuristic-Based Collision Resolution**  
  Functions such as **`fix_direct_collisions_time`**, **`fix_calulated_collisions_time`**, and **`fix_near_misses_time`** are designed to apply specific heuristics when a conflict is detected. They:
  - Identify the collision or near miss.
  - Evaluate drone battery levels to determine which drone is better suited for a schedule adjustment.
  - Use the above mechanisms (e.g., delaying transit times) to modify the flight plan and resolve the conflict.

- **Planning Adjustment and Generation**  
  - **`make_new_planning(planning: Dict[str, pd.DataFrame], drone_speed: int, charging_station_position: tuple, threshold: int, time_step: float)`**  
    Generates a new planning solution by first detecting collisions and near misses, then selecting and applying an appropriate heuristic. This iterative approach is crucial for evolving the planning towards a collision-free state.


## Config parameters

This configuration file provides various parameters required for the simulated annealing process. It contains multiple configuration sets, each identified by a unique key (e.g., `config-0`, `config-A`, `config-B`, etc.). Each configuration includes the following parameters:

- **`collision_penalty`**:  
  A numerical value representing the cost multiplier applied for each collision detected. A higher value increases the penalty for collisions.

- **`avoidance_penalty`**:  
  A numerical value representing the penalty for near misses or situations requiring avoidance maneuvers. This value influences how aggressively the algorithm avoids potential collisions.

- **`total_duration_penalty`**:  
  A multiplier applied to the total flight duration to account for the efficiency of the overall scheduling. It balances the trade-off between flight time and collision avoidance.

- **`T_init`** (Initial Temperature):  
  The starting temperature for the simulated annealing algorithm. A higher initial temperature allows the algorithm to explore a broader range of solutions.

- **`T_freeze`** (Freeze Temperature):  
  The temperature at which the algorithm stops iterating. This value determines the stopping criterion for the annealing process.

- **`alpha_T`** (Cooling Rate):  
  The rate at which the temperature decreases. A value closer to 1 results in slower cooling, allowing more exploration per temperature level.

- **`k_iter`** (Iterations per Temperature Step):  
  The number of iterations or modifications performed at each temperature level before the temperature is reduced.

These parameters can be tuned to adjust the behavior of the simulated annealing process, balancing the exploration of new solutions against the convergence toward an optimal planning configuration.

## Dependencies
To run the scripts, install the following Python packages:
```bash
pip install numpy pandas shapely tqdm matplotlib
```

## Notes

- This module is designed to be integrated into a larger project and does not include a standalone execution script.

- Drone trajectory data should be provided as a dictionary of Pandas DataFrames, where each key represents a drone and the corresponding value contains its flight plan.
