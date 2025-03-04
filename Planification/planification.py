import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from Planification.warehouse import Warehouse3D
from networkx.classes import is_empty
import json


def open_csv(csv_file_name: str) -> pd.DataFrame:
    """Open a CSV file and return a pandas DataFrame."""
    path = os.path.join(os.path.dirname(__file__), "TaskList", csv_file_name)
    return pd.read_csv(path, header=0)

def assign_drones(csv_file: pd.DataFrame, num_drones: int) -> pd.DataFrame:
    """Assign drones to tasks in a cyclic manner."""
    csv_file['drone'] = [f'd{i % num_drones + 1}' for i in range(len(csv_file))]
    return csv_file

def compute_distance(bellman_matrix: np.ndarray, position_dict: Dict[Tuple[int, int, int], int],
                    start: list, end: list) -> int:
    """Compute the distance between two points using the Bellman matrix."""
    n1 = position_dict[tuple(json.loads(start))] - 1
    n2 = position_dict[tuple(json.loads(end))] - 1

    return bellman_matrix[n1][n2]


def compute_travel_time(start: Tuple[int, int, int], end: Tuple[int, int, int], drone_speed : int) -> timedelta:
    """Compute the travel time between two points."""
    distance = abs(start[0] - end[0]) + abs(start[1] - end[1]) + abs(start[2] - end[2])
    time = distance / drone_speed
    return timedelta(minutes=time)

def prepare_csv(bellman_matrix: np.ndarray, position_dict: Dict[Tuple[int, int, int], int],
                csv_file_name: str, warehouse: Warehouse3D, num_drones: int, drone_speed : int) -> pd.DataFrame:
    """Prepare the CSV file by sorting, assigning drones, and computing distances."""
    csv_file = open_csv(csv_file_name)
    csv_file['time'] = pd.to_datetime(csv_file['time'], format='%H:%M:%S').dt.strftime('%H:%M:%S')
    csv_file = csv_file.sort_values(by=['time'])
    csv_file = assign_drones(csv_file, num_drones)

    # csv_file['task_distance'] = csv_file.apply(
    #     lambda row: compute_distance(bellman_matrix, position_dict, row['pos0'],row['pos1']), axis=1)

    csv_file[['task_distance_2', 'task_path']] = csv_file.apply(
        lambda row: pd.Series(warehouse.compute_manhattan_distance_with_BFS(
            tuple(json.loads(row['pos0'])),
            tuple(json.loads(row['pos1'])),
            return_path=True,
            reduced=True
        )),
        axis=1
    )

    # Vérification des valeurs infinies
    if np.isinf(csv_file['task_distance_2']).any():
        raise ValueError(
            "Error: One distance or more are infinite. Check configuration")

    # csv_file['task_time'] = csv_file['task_distance'] / drone_speed
    # csv_file['task_time'] = csv_file['task_time'].apply(lambda x: timedelta(minutes=x))

    csv_file['task_time_2'] = csv_file['task_distance_2'] / drone_speed
    csv_file['task_time_2'] = csv_file['task_time_2'].apply(lambda x: timedelta(minutes=x))


    return csv_file


def create_drone_schedule(csv: pd.DataFrame, drone_number: int, warehouse: Warehouse3D, drone_speed : int, task_nb_before_charge : int) -> pd.DataFrame:
    """Create a full schedule for a specific drone."""

    object_time_treatment = timedelta(seconds=30)
    charge_time= timedelta(minutes=30)

    new_time = csv.loc[csv['drone'] == f'd{drone_number}']['time'].iloc[0]
    new_time = datetime.strptime(new_time, "%H:%M:%S").time()

    df_path = pd.DataFrame(columns=['time', 'task_type', 'id', 'position'])
    first_iteration = True

    task_count = 0

    for _, row in csv.loc[csv['drone'] == f'd{drone_number}', ['task_path', 'task_type', 'id', 'time']].iterrows():

        task_id = row['id']
        task_type = row['task_type']
        positions = row['task_path']
        time_start = datetime.strptime(row['time'], "%H:%M:%S").time()

        if task_count == task_nb_before_charge:
            task_count = 0
            cc = warehouse.get_charge_location()
            __, return_charge_path = warehouse.compute_manhattan_distance_with_BFS(df_path.iloc[-1]['position'], cc,
                                                                                   True, True)
            for pos in return_charge_path:
                time_delta = compute_travel_time(pos, df_path.iloc[-1]['position'], drone_speed)
                new_time = (datetime.combine(datetime.today(), new_time) + time_delta).time()
                df_path.loc[len(df_path)] = [new_time, "RC", "", pos]
            new_time = (datetime.combine(datetime.today(), new_time) + charge_time).time()


        if time_start > new_time:
            new_time = time_start

        if not first_iteration:

            __, repositioning_path = warehouse.compute_manhattan_distance_with_BFS(df_path.iloc[-1]['position'], positions[0], True, True)
            for pos in repositioning_path:
                time_delta = compute_travel_time(pos, df_path.iloc[-1]['position'], drone_speed)
                new_time = (datetime.combine(datetime.today(), new_time) + time_delta).time()
                df_path.loc[len(df_path)] = [new_time, "PT","", pos]

        for pos in positions:
            if task_type == "A" or task_type == "D":
                new_time = (datetime.combine(datetime.today(), new_time) + object_time_treatment).time()
            if df_path.empty:
                df_path.loc[len(df_path)] = [new_time, task_type, task_id, pos]
            else:
                time_delta = compute_travel_time(pos, df_path.iloc[-1]['position'], drone_speed)
                new_time = (datetime.combine(datetime.today(), new_time) + time_delta).time()
                df_path.loc[len(df_path)] = [new_time, task_type, task_id, pos]

        new_time = (datetime.combine(datetime.today(), new_time) + object_time_treatment).time()
        first_iteration = False
        task_count += 1

    return df_path

def schedule(bellman_matrix: np.ndarray, position_dict: Dict[Tuple[int, int, int], int],
                      warehouse: Warehouse3D, num_drones: int, drone_speed :int) -> Dict[str, pd.DataFrame]:
    """Prepare the schedule for all drones."""
    task_nb_before_charge = 3
    csv_file_name = f'TL_{warehouse.name}.csv'
    csv = prepare_csv(bellman_matrix, position_dict, csv_file_name, warehouse, num_drones, drone_speed)
    drone_schedules = {f'd{i}': create_drone_schedule(csv, i, warehouse, drone_speed, task_nb_before_charge) for i in range(1, num_drones + 1)}

    return drone_schedules