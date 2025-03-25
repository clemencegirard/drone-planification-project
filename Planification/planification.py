import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from Warehouse.warehouse import Warehouse3D
import json
from pathlib import Path


def load_config_planning( planning_name : str, config_path="config_planning.json"):
    config_file = Path(__file__).parent / config_path
    with open(config_file, "r") as f:
        config = json.load(f)

    planning_config = config["planning"][planning_name]
    mapping_config = config["task_name_mapping"]

    return planning_config, mapping_config


def open_csv(csv_file_name: str) -> pd.DataFrame:
    """Open a CSV file and return a pandas DataFrame."""
    path = os.path.join(os.path.dirname(__file__), "TaskList", csv_file_name)
    return pd.read_csv(path, header=0)


def compute_travel_time(start: Tuple[int, int, int], end: Tuple[int, int, int], drone_speed: int,
                        distance: Optional[int] = None) -> timedelta:
    """
    Compute the travel time between two points in the warehouse.

    Args:
        start: Starting position as a tuple (x, y, z).
        end: Ending position as a tuple (x, y, z).
        drone_speed: Speed of the drone in units per minute.
        distance: Optional precomputed distance between start and end.

    Returns:
        Travel time as a timedelta.
    """
    if distance is None:
        # If no distance is provided, compute the Manhattan distance between start and end
        distance = abs(start[0] - end[0]) + abs(start[1] - end[1]) + abs(start[2] - end[2])

    # Compute travel time
    time = distance / drone_speed
    return timedelta(minutes=time)


def prepare_csv(csv_file_name: str, warehouse: Warehouse3D , config: json) -> pd.DataFrame:
    """
    Prepare the CSV file by sorting, assigning drones, and computing distances.

    Args:
        csv_file_name: Name of the CSV file containing the tasks.
        warehouse: The warehouse object.
        config : json parameters of the planning.

    Returns:
        A DataFrame with the prepared tasks.
    """
    drone_speed = config["drone_speed"]

    csv_file = open_csv(csv_file_name)
    csv_file['time'] = pd.to_datetime(csv_file['time'], format='%H:%M:%S').dt.strftime('%H:%M:%S')
    csv_file = csv_file.sort_values(by=['time'])

    # Compute task distances and paths
    csv_file[['task_distance', 'task_path']] = csv_file.apply(
        lambda row: pd.Series(warehouse.compute_manhattan_distance_with_BFS(
            tuple(json.loads(row['pos0'])),
            tuple(json.loads(row['pos1'])),
            return_path=True,
            reduced=True
        )),
        axis=1
    )

    # Compute task distances and paths
    csv_file[['task_distance', 'task_path']] = csv_file.apply(
        lambda row: pd.Series(warehouse.compute_manhattan_distance_with_BFS(
            tuple(json.loads(row['pos0'])),
            tuple(json.loads(row['pos1'])),
            return_path=True,
            reduced=True
        )),
        axis=1
    )



    # Check for infinite distances
    if np.isinf(csv_file['task_distance']).any():
        raise ValueError("Error: One or more distances are infinite. Check configuration.")

    # Compute return-to-charge distances and paths
    csv_file[['rc_distance', 'rc_path']] = csv_file.apply(
        lambda row: pd.Series(warehouse.compute_manhattan_distance_with_BFS(
            tuple(json.loads(row['pos1'])),
            tuple(warehouse.get_charge_location()),
            return_path=True,
            reduced=True
        )),
        axis=1
    )

    # Compute full distance and time
    csv_file['full_distance'] = csv_file['task_distance'] + csv_file['rc_distance']
    csv_file['full_time'] = csv_file['full_distance'] / drone_speed
    csv_file['full_time'] = csv_file['full_time'].apply(lambda x: timedelta(minutes=x))
    csv_file['Done'] = False

    return csv_file


def full_task_treatment(drone_time_follow: Dict[str, datetime], drone_nb: str, start_time: Optional[datetime],
                        planning: Dict[str, pd.DataFrame], task_path: List[Tuple[int, int, int]],
                        warehouse: Warehouse3D, drone_last_position: Dict[str, Tuple[int, int, int]],
                        drone_speed: int, drone_battery: Dict[str, timedelta], task_type: str,
                        object_time_treatment: timedelta, rc_path: List[Tuple[int, int, int]], task_id: str,
                        csv: pd.DataFrame, index_task: int, max_fly_time : timedelta, mapping_config : json) -> Dict[str, pd.DataFrame]:
    """
    Handle the full treatment of a task, including positioning, task processing, and returning to charge.

    Args:
        drone_time_follow: Dictionary tracking the next available time for each drone.
        drone_nb: The drone identifier.
        start_time: The start time of the task.
        planning: Dictionary of DataFrames representing the schedule for each drone.
        task_path: The path for the task.
        warehouse: The warehouse object.
        drone_last_position: Dictionary tracking the last position of each drone.
        drone_speed: Speed of the drone.
        drone_battery: Dictionary tracking the battery level of each drone.
        task_type: The type of task (e.g., 'A', 'D').
        object_time_treatment: Time required to process the task.
        rc_path: The path to return to the charging station.
        task_id: The task identifier.
        csv: The DataFrame containing all tasks.
        index_task: The index of the task in the DataFrame.
        max_fly_time : Maximal fly time
        mapping_config : Config json for task naming

    Returns:
        Updated planning dictionary.
    """
    if start_time is not None:
        drone_time_follow[drone_nb] = start_time

    # Position the drone at the start of the task
    positioning_time, planning = positionning(planning, drone_nb, warehouse, drone_last_position[drone_nb],
                                              task_path[0], drone_speed, drone_time_follow, drone_battery, max_fly_time, mapping_config,True)

    drone_last_position[drone_nb] = task_path[-1]  # Update the drone's final position


    #verif si le drone a deja été bougé ou non
    if planning[drone_nb].empty:
        # First position, the drone starts at 'first_pos'
        inital_pos = warehouse.get_charge_location()
    else:
        # Last position in the DataFrame to determine the starting position
        inital_pos = planning[drone_nb].iloc[-1]['position']


    # Process the task
    planning = task_processing(task_type, drone_time_follow, drone_nb, task_path, object_time_treatment, planning,
                               rc_path, task_id, drone_speed, drone_battery, max_fly_time, mapping_config, inital_pos)

    # Return the drone to the charging station
    drone_last_position[drone_nb] = rc_path[-1]

    if type(index_task) != int:
        index_task = index_task[0]

    # Mark the task as done
    csv.loc[(csv.index == index_task) & (csv["task_type"] == task_type), 'Done'] = True

    return planning


def task_processing(task_type: str, drone_time_follow: Dict[str, datetime], drone_nb: str,
                    task_path: List[Tuple[int, int, int]], object_time_treatment: timedelta,
                    planning: Dict[str, pd.DataFrame], rc_path: List[Tuple[int, int, int]],
                    task_id: str, drone_speed: int, drone_battery: Dict[str, timedelta], max_fly_time : timedelta, mapping_config : json, inital_pos : Tuple[int, int, int] ) -> Dict[str, pd.DataFrame]:
    """
    Process the task and update the drone's schedule.

    Args:
        task_type: The type of task (e.g., 'A', 'D').
        drone_time_follow: Dictionary tracking the next available time for each drone.
        drone_nb: The drone identifier.
        task_path: The path for the task.
        object_time_treatment: Time required to process the task.
        planning: Dictionary of DataFrames representing the schedule for each drone.
        rc_path: The path to return to the charging station.
        task_id: The task identifier.
        drone_speed: Speed of the drone.
        drone_battery: Dictionary tracking the battery level of each drone.
        mapping_config : Config json for task naming
        inital_pos: Tuple of the first position of the drone (usualy charger)

    Returns:
        Updated planning dictionary.
    """


    if task_type == mapping_config['arrival'] or task_type == mapping_config['departure']:
        drone_time_follow[drone_nb] = drone_time_follow[drone_nb] + object_time_treatment
        drone_battery[drone_nb] -= object_time_treatment

    # Add the task path to the drone's schedule
    planning[drone_nb] = add_path_to_df(planning[drone_nb], task_path, task_type, task_id, drone_speed,
                                        drone_time_follow, drone_nb, inital_pos, drone_battery, max_fly_time)

    if task_type == mapping_config['arrival'] or task_type == mapping_config['departure']:
        drone_time_follow[drone_nb] = drone_time_follow[drone_nb] + object_time_treatment
        drone_battery[drone_nb] -= object_time_treatment

    # Add the return-to-charge path to the drone's schedule
    planning[drone_nb] = add_path_to_df(planning[drone_nb], rc_path, mapping_config['return_charge'], "", drone_speed, drone_time_follow, drone_nb,
                                        inital_pos, drone_battery, max_fly_time)

    return planning


def positionning(planning: Dict[str, pd.DataFrame], drone_nb: str, warehouse: Warehouse3D,
                 start: Tuple[int, int, int], end: Tuple[int, int, int], drone_speed: int,
                 drone_time_follow: Dict[str, datetime], drone_battery: Dict[str, timedelta], max_fly_time : timedelta, mapping_config : json,
                 allow_modification: bool = False) -> Tuple[timedelta, Dict[str, pd.DataFrame]]:
    """
    Position the drone at the start of the task.

    Args:
        planning: Dictionary of DataFrames representing the schedule for each drone.
        drone_nb: The drone identifier.
        warehouse: The warehouse object.
        start: Starting position as a tuple (x, y, z).
        end: Ending position as a tuple (x, y, z).
        drone_speed: Speed of the drone.
        drone_time_follow: Dictionary tracking the next available time for each drone.
        drone_battery: Dictionary tracking the battery level of each drone.
        max_fly_time : Max fly time
        allow_modification: Whether to allow modification of the planning.
        mapping_config : Config json for task naming

    Returns:
        A tuple containing the positioning time and the updated planning.
    """
    positionning_dist, positionning_path = warehouse.compute_manhattan_distance_with_BFS(start, end, True, True)
    time = compute_travel_time(None, None, drone_speed, positionning_dist)

    if allow_modification:
        planning[drone_nb] = add_path_to_df(planning[drone_nb], positionning_path, mapping_config['positioning'], '', drone_speed,
                                            drone_time_follow, drone_nb, start, drone_battery, max_fly_time)

    return time, planning


def add_path_to_df(df: pd.DataFrame, positions: List[Tuple[int, int, int]], task_type: str,
                   task_id: str, drone_speed: int, drone_time_follow: Dict[str, datetime],
                   drone_nb: str, first_pos: Tuple[int, int, int], drone_battery: Dict[str, timedelta], max_fly_time : timedelta) -> pd.DataFrame:
    """
    Add a path to the drone's schedule DataFrame.

    Args:
        df: The DataFrame representing the drone's schedule.
        positions: List of positions in the path.
        task_type: The type of task (e.g., 'A', 'D').
        task_id: The task identifier.
        drone_speed: Speed of the drone.
        drone_time_follow: Dictionary tracking the next available time for each drone.
        drone_nb: The drone identifier.
        first_pos: The first position in the path.
        drone_battery: Dictionary tracking the battery level of each drone.
        max_fly_time : Maximal fly time

    Returns:
        Updated DataFrame with the new path added.
    """
    for pos in positions:

        # verif si le drone a deja été bougé ou non
        if not df.empty:
            first_pos = df.iloc[-1]['position']

        # Compute travel time
        time_delta = compute_travel_time(first_pos, pos, drone_speed)

        # Check if drone_time_follow[drone_nb] is already a datetime
        if isinstance(drone_time_follow[drone_nb], timedelta):
            # If it's a timedelta, add it to a reference datetime (midnight) and update it
            drone_time_follow[drone_nb] = datetime.combine(datetime.today(), datetime.min.time()) + drone_time_follow[drone_nb]

        time = drone_time_follow[drone_nb] + time_delta  # Direct addition
        drone_time_follow[drone_nb] = time

        drone_battery[drone_nb] -= time_delta

        # Add a new row to the DataFrame with the new position and time
        battery_percentage =  round((drone_battery[drone_nb]/max_fly_time)*100,2)

        df.loc[len(df)] = [time, task_type, task_id, pos, drone_battery[drone_nb], battery_percentage]

    return df


def drone_initial_selection(drone_last_position: Dict[str, Tuple[int, int, int]],
                            drone_time_follow: Dict[str, datetime], drone_battery: Dict[str, timedelta],
                            row: pd.Series, warehouse: Warehouse3D, drone_speed: int,
                            object_time_treatment: timedelta, lower_threshold : float) -> Tuple[Optional[datetime], Optional[str]]:
    """
    Select the best drone for a task based on battery level and availability.

    Args:
        drone_last_position: Dictionary tracking the last position of each drone.
        drone_time_follow: Dictionary tracking the next available time for each drone.
        drone_battery: Dictionary tracking the battery level of each drone.
        row: The row from the DataFrame representing the task.
        warehouse: The warehouse object.
        drone_speed: Speed of the drone.
        object_time_treatment: Time required to process the task.
        lower_threshold : Lower battery threshold.

    Returns:
        A tuple containing the start time and the selected drone identifier, or (None, None) if no drone is available.
    """
    best_drone = None
    best_time = timedelta.max
    full_time = row['full_time'] + 2 * object_time_treatment
    task_start_time = datetime.strptime(row['time'], '%H:%M:%S').time()
    task_start_time = datetime.combine(datetime.today(), task_start_time)
    task_path_start = row['task_path'][0]

    for drone_nb, last_pos in drone_last_position.items():
        # Compute travel time to the task
        total_dist_position_path, position_path = warehouse.compute_manhattan_distance_with_BFS(last_pos,task_path_start, True,True)
        time_to_start = compute_travel_time(None, None, drone_speed, total_dist_position_path)

        # Check battery availability
        battery_required = time_to_start.total_seconds()  # Consumption during travel
        total_battery_required = battery_required + full_time.total_seconds()

        drone_battery_seconds = drone_battery[drone_nb].total_seconds() if isinstance(drone_battery[drone_nb],timedelta) else drone_battery[drone_nb]

        if drone_battery_seconds >= total_battery_required and (drone_battery_seconds - total_battery_required) >= lower_threshold*drone_battery_seconds:
            # Determine the next availability of the drone
            drone_available_time = max(drone_time_follow[drone_nb], task_start_time) if drone_time_follow[drone_nb] else task_start_time

            # Convert to timedelta for comparison
            task_start_delta = timedelta(hours=task_start_time.hour, minutes=task_start_time.minute,
                                         seconds=task_start_time.second)
            drone_available_delta = timedelta(hours=drone_available_time.hour, minutes=drone_available_time.minute,
                                              seconds=drone_available_time.second)

            # Compute the real start time
            real_start_time = max(task_start_delta, drone_available_delta)

            # Compute arrival and finish time
            total_time = real_start_time + time_to_start + full_time
            arrival_time = (datetime.combine(datetime.today(), datetime.min.time()) + total_time).time()
            arrival_time_delta = timedelta(hours=arrival_time.hour, minutes=arrival_time.minute,
                                           seconds=arrival_time.second)

            # Select the drone that finishes the earliest
            if arrival_time_delta < best_time:
                best_drone = drone_nb
                best_time = arrival_time_delta

    # If no drone is selected after testing all, return None, None
    if best_drone is None:
        return None, None

    return real_start_time, best_drone


def drone_empty_battery_selection(drone_quantity: int, csv: pd.DataFrame, drone_battery: Dict[str, timedelta],
                                  planning: Dict[str, pd.DataFrame], warehouse: Warehouse3D,
                                  drone_last_position: Dict[str, Tuple[int, int, int]], drone_speed: int,
                                  drone_time_follow: Dict[str, datetime], charge_speed: int,
                                  object_time_treatment: timedelta, lower_threshold : float, max_fly_time : timedelta, mapping_config : json) -> Tuple[
    Dict[str, datetime], Dict[str, str], Dict[str, str], Dict[str, timedelta], Dict[str, timedelta]]:
    """
    Select tasks for drones with empty batteries and compute charging times.

    Args:
        drone_quantity: Number of drones available.
        csv: The DataFrame containing all tasks.
        drone_battery: Dictionary tracking the battery level of each drone.
        planning: Dictionary of DataFrames representing the schedule for each drone.
        warehouse: The warehouse object.
        drone_last_position: Dictionary tracking the last position of each drone.
        drone_speed: Speed of the drone.
        drone_time_follow: Dictionary tracking the next available time for each drone.
        charge_speed: Charging speed of the drones.
        object_time_treatment: Time required to process the task.
        lower_threshold : Lower battery threshold
        max_fly_time: Maximal fly time
        mapping_config: Mapping config of param name

    Returns:
        A tuple containing dictionaries for best finish times, best tasks, best task types, time for charge, and battery gain.
    """
    best_finish_time = {f'd{i}': None for i in range(1, drone_quantity + 1)}
    best_task = {f'd{i}': None for i in range(1, drone_quantity + 1)}
    best_task_type = {f'd{i}': None for i in range(1, drone_quantity + 1)}
    battery_gain = {f'd{i}': None for i in range(1, drone_quantity + 1)}
    time_for_charge = {f'd{i}': None for i in range(1, drone_quantity + 1)}

    # Get unassigned tasks sorted by time
    unassigned_tasks = csv.loc[(csv['Done'] == False)].sort_values('time').copy()

    # Create a set for tracking assignable tasks
    assignable_tasks = set(unassigned_tasks['id'].tolist())
    for drone in drone_battery.keys():
        min_finish_time = None
        best_task_for_drone = None
        best_task_type_for_drone = None
        best_delta = None
        best_battery_gain = None

        # Iterate through tasks in chronological order
        for index2, row2 in unassigned_tasks[unassigned_tasks['id'].isin(assignable_tasks)].iterrows():
            task_id2 = row2['id']
            task_type2 = row2['task_type']
            task_path2 = row2['task_path']
            task_time = datetime.strptime(row2['time'], '%H:%M:%S').time()
            task_start_time2 = datetime.combine(datetime.today(), task_time)

            # Compute parameters
            positioning_time2, __ = positionning(planning, drone, warehouse, drone_last_position[drone], task_path2[0],drone_speed, drone_time_follow, drone_battery, max_fly_time, mapping_config, False)
            full_time2 = row2['full_time'] + positioning_time2 + 2 * object_time_treatment

            # battery_threshold_test(full_time2,drone_battery[drone],max_fly_time,lower_threshold)

            # Compute cumulative time with priority to start time
            potential_start = max(task_start_time2, drone_time_follow[drone])
            delta, battery_gain_value = delta_battery(drone_battery[drone], charge_speed, full_time2, max_fly_time, lower_threshold)
            potential_cumulated_time = potential_start + delta + full_time2

            # Initial selection of the first encountered task
            if best_task_for_drone is None:
                min_finish_time = potential_cumulated_time
                best_task_for_drone = task_id2
                best_task_type_for_drone = task_type2
                best_delta = delta
                best_battery_gain = battery_gain_value
                continue  # Skip to the next iteration to compare other tasks

            # Compare with the already selected task
            existing_time_str = csv.loc[csv['id'] == best_task_for_drone, 'time'].iloc[0]
            existing_time = datetime.strptime(existing_time_str, '%H:%M:%S').time()
            existing_time = datetime.combine(datetime.today(), existing_time)

            # Double selection criteria: task time FIRST, then cumulative time
            is_earlier = task_start_time2 < existing_time
            same_time_better = (task_start_time2 == existing_time) and (potential_cumulated_time < min_finish_time)

            if is_earlier or same_time_better:
                min_finish_time = potential_cumulated_time
                best_task_for_drone = task_id2
                best_task_type_for_drone = task_type2
                best_delta = delta
                best_battery_gain = battery_gain_value

        # Final assignment
        if best_task_for_drone is not None:
            best_finish_time[drone] = min_finish_time
            best_task[drone] = best_task_for_drone
            best_task_type[drone] = best_task_type_for_drone
            time_for_charge[drone] = best_delta
            battery_gain[drone] = best_battery_gain
            assignable_tasks.remove(best_task_for_drone)

    return best_finish_time, best_task, best_task_type, time_for_charge, battery_gain



def delta_battery(current_level: timedelta, charge_speed: int, target_level: timedelta,
                  max_fly_time: timedelta = None, lower_threshold: float = None) -> Tuple[timedelta, timedelta]:
    """
    Compute the time and battery gain required to reach the target battery level.

    Args:
        current_level: Current battery level.
        charge_speed: Charging speed of the drone.
        target_level: Target battery level.
        max_fly_time: Maximal fly time.
        lower_threshold: Lower threshold battery.

    Returns:
        A tuple containing the time needed and the battery gain.
    """
    extra_battery = timedelta(seconds=0)
    if max_fly_time is not None and lower_threshold is not None:
        extra_battery = max_fly_time * lower_threshold  # Évite l'erreur si l'un des deux est None

    # Nouvelle cible = niveau nécessaire pour la mission + batterie requise pour ne pas descendre sous le seuil
    adjusted_target_level = target_level + extra_battery

    if current_level < adjusted_target_level:
        battery_gain = adjusted_target_level - current_level
        time_needed = battery_gain / charge_speed
        return time_needed, battery_gain
    else:
        return timedelta(seconds=0), timedelta(seconds=0)




def drone_charging_process(drone_time_follow: Dict[str, datetime], drone_battery: Dict[str, timedelta],
                           battery_gain: Dict[str, timedelta], time_for_charge: Dict[str, timedelta],
                           max_fly_time: timedelta, upper_threshold : float, drone_nb: str, specific_drone: bool = False) -> Tuple[
    Dict[str, datetime], Dict[str, timedelta]]:
    """
    Handle the charging process for drones.

    Args:
        drone_time_follow: Dictionary tracking the next available time for each drone.
        drone_battery: Dictionary tracking the battery level of each drone.
        battery_gain: Dictionary tracking the battery gain for each drone.
        time_for_charge: Dictionary tracking the time required for charging for each drone.
        max_fly_time: Maximum flying time for the drones.
        upper_threshold: Upper bound of max recommanded charge
        drone_nb: The drone identifier.
        specific_drone: Whether to charge a specific drone or all drones.

    Returns:
        Updated dictionaries for drone_time_follow and drone_battery.
    """
    if not specific_drone:
        for drone in drone_time_follow.keys():
            if time_for_charge[drone] is not None:
                drone_time_follow[drone] += time_for_charge[drone]
                if drone_battery[drone] + battery_gain[drone] < max_fly_time:
                    drone_battery[drone] += battery_gain[drone]
                else:
                    drone_battery[drone] = upper_threshold*max_fly_time
    else:
        if time_for_charge != timedelta(seconds=0):
            drone_time_follow[drone_nb] += time_for_charge
            if drone_battery[drone_nb] + battery_gain < max_fly_time:
                drone_battery[drone_nb] += battery_gain
            else:
                drone_battery[drone_nb] = upper_threshold*max_fly_time

    return drone_time_follow, drone_battery


def create_drone_schedule(csv: pd.DataFrame, warehouse: Warehouse3D, config : json, mapping_config : json) -> Dict[str, pd.DataFrame]:
    """
    Create a full schedule for a fleet of drones.

    Args:
        csv: The DataFrame containing all tasks.
        warehouse: The warehouse object.
        config : Config json of the parameters of the planning
        mapping_config : Config json for task naming

    Returns:
        A dictionary of DataFrames representing the schedule for each drone.
    """

    ## Config Parameters ##
    drone_quantity = config["drone_quantity"]
    drone_speed = config["drone_speed"]
    object_time_treatment = timedelta(seconds=config["object_time_treatment"])
    max_fly_time = timedelta(seconds=config["battery"]["max_capacity"])
    charge_speed = config["battery"]["charger_speed"]
    battery_initial_charge =  timedelta(seconds=config["battery"]["initial_charge"])
    lower_threshold = config["battery"]["lower_threshold"]
    upper_threshold = config["battery"]["upper_threshold"]

    ## Dict Creation ##
    drone_battery = {f'd{i}': battery_initial_charge for i in range(1, drone_quantity + 1)}
    drone_last_position = {f'd{i}': warehouse.get_charge_location() for i in range(1, drone_quantity + 1)}
    planning = {f'd{i}': pd.DataFrame(columns=['time', 'task_type', 'id', 'position', 'battery_time', 'battery_percentage']) for i in range(1, drone_quantity + 1)}
    drone_time_follow = {f'd{i}': None for i in range(1, drone_quantity + 1)}

    ## Scheduling Algorithm ##
    while True:
        # Refresh the list of uncompleted tasks at each iteration
        undone_tasks = csv.loc[(csv['Done'] == False)]

        if undone_tasks.empty:
            break

        for index, row in undone_tasks.iterrows():
            # Check again if the task is still uncompleted
            if csv.at[index, 'Done']:
                continue

            task_id = row['id']
            task_type = row['task_type']
            task_path = row['task_path']
            rc_path = row['rc_path']
            task_start_time = datetime.strptime(row['time'], '%H:%M:%S').time()

            # Select the best drone for the task
            start_time, drone_nb = drone_initial_selection(drone_last_position, drone_time_follow, drone_battery, row, warehouse, drone_speed, object_time_treatment, lower_threshold)

            if not drone_nb:  # If no drone has enough battery for the current task
                best_finish_time, best_task, best_task_type, time_for_charge, battery_gain = drone_empty_battery_selection(drone_quantity, csv, drone_battery, planning, warehouse, drone_last_position, drone_speed,drone_time_follow, charge_speed, object_time_treatment, lower_threshold,max_fly_time,mapping_config)
                drone_time_follow, drone_battery = drone_charging_process(drone_time_follow, drone_battery,battery_gain, time_for_charge, max_fly_time, upper_threshold,"")

                for drone in drone_time_follow.keys():  # Assign tasks to drones
                    if best_task[drone] is None:
                        continue

                    task_type_2 = best_task_type[drone]

                    # Filter tasks based on ID and type
                    task_filter = (csv['id'] == best_task[drone]) & (csv['task_type'] == task_type_2)

                    task_path_2 = csv.loc[task_filter, 'task_path'].iloc[0]
                    rc_path_2 = csv.loc[task_filter, 'rc_path'].iloc[0]
                    task_id_2 = csv.loc[task_filter, 'id'].iloc[0]
                    index_2 = csv.loc[task_filter].index

                    planning = full_task_treatment(drone_time_follow, drone, None, planning, task_path_2, warehouse,
                                                   drone_last_position, drone_speed, drone_battery, task_type_2,
                                                   object_time_treatment, rc_path_2, task_id_2, csv, index_2, max_fly_time, mapping_config)

            else:  # If a drone is available in terms of battery
                start_time_absolute = datetime.combine(datetime.today().date(), task_start_time)

                if drone_time_follow[drone_nb] is not None and drone_time_follow[drone_nb] < start_time_absolute:  # If there is time to charge before the task starts
                    time_for_charge, battery_gain = delta_battery(drone_time_follow[drone_nb], charge_speed, start_time_absolute)
                    drone_time_follow, drone_battery = drone_charging_process(drone_time_follow, drone_battery, battery_gain, time_for_charge, max_fly_time, upper_threshold, drone_nb, True)

                planning = full_task_treatment(drone_time_follow, drone_nb, start_time, planning, task_path, warehouse,
                                               drone_last_position, drone_speed, drone_battery, task_type,
                                               object_time_treatment, rc_path, task_id, csv, index, max_fly_time, mapping_config)

    return planning


def schedule(warehouse: Warehouse3D, config : json, mapping_config :json) -> Dict[str, pd.DataFrame]:
    """
    Prepare the schedule for all drones.

    Args:
        warehouse: The warehouse object.
        config: Config json of parameters of the planning
        mapping_config : Config json for task naming

    Returns:
        A dictionary of DataFrames representing the schedule for each drone.
    """
    csv_file_name = f'TL_{warehouse.name}.csv'
    csv = prepare_csv(csv_file_name, warehouse, config)
    drone_schedules = create_drone_schedule(csv, warehouse, config, mapping_config)

    return drone_schedules