import os
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from warehouse import Warehouse3D


def open_csv(csv_file_name: str) -> pd.DataFrame:
    """Open a CSV file and return a pandas DataFrame."""
    path = os.path.join(os.path.dirname(__file__), "TaskList", csv_file_name)
    return pd.read_csv(path, header=0)


def assign_drones(csv_file: pd.DataFrame, num_drones: int) -> pd.DataFrame:
    """Assign drones to tasks in a cyclic manner."""
    csv_file['drone'] = [f'd{i % num_drones + 1}' for i in range(len(csv_file))]
    return csv_file


def compute_distance(bellman_matrix: np.ndarray, position_dict: Dict[Tuple[int, int, int], int],
                    start: List[int], end: List[int]) -> int:
    """Compute the distance between two points using the Bellman matrix."""
    n1 = position_dict[tuple(start)] - 1
    n2 = position_dict[tuple(end)] - 1
    return bellman_matrix[n1][n2]


def compute_travel_time(start: Tuple[int, int, int], end: Tuple[int, int, int]) -> timedelta:
    """Compute the travel time between two points."""
    return timedelta(minutes=abs(start[0] - end[0]) + abs(start[1] - end[1]) + abs(start[2] - end[2]))


def prepare_csv(bellman_matrix: np.ndarray, position_dict: Dict[Tuple[int, int, int], int],
                csv_file_name: str, warehouse: Warehouse3D, num_drones: int) -> pd.DataFrame:
    """Prepare the CSV file by sorting, assigning drones, and computing distances."""
    csv_file = open_csv(csv_file_name)
    csv_file['time'] = pd.to_datetime(csv_file['time'], format='%H:%M:%S').dt.strftime('%H:%M:%S')
    csv_file = csv_file.sort_values(by=['time'])
    csv_file = assign_drones(csv_file, num_drones)

    csv_file['task_time'] = csv_file.apply(
        lambda row: compute_distance(bellman_matrix, position_dict, [row['row0'], row['col0'], row['height0']],
                                     [row['row1'], row['col1'], row['height1']]),
        axis=1
    )

    csv_file[['task_time_2', 'task_path']] = csv_file.apply(
        lambda row: pd.Series(warehouse.compute_manhattan_distance_with_BFS(
            [row['row0'], row['col0'], row['height0']],
            [row['row1'], row['col1'], row['height1']],
            True
        )),
        axis=1
    )
    return csv_file


def create_drone_schedule(csv: pd.DataFrame, drone_number: int, warehouse: Warehouse3D) -> pd.DataFrame:
    """Create a full schedule for a specific drone."""
    new_time = csv.loc[csv['drone'] == f'd{drone_number}']['time'].iloc[0]
    new_time = datetime.strptime(new_time, "%H:%M:%S").time()
    df_path = pd.DataFrame(columns=['time', 'task_type', 'id', 'position'])
    last_state = None
    first_iteration = True
    object_time_treatment = timedelta(seconds=30)

    for _, row in csv.loc[csv['drone'] == f'd{drone_number}', ['task_path', 'task_type', 'id', 'time']].iterrows():
        task_id = row['id']
        task_type = row['task_type']
        positions = row['task_path']
        time_start = datetime.strptime(row['time'], "%H:%M:%S").time()

        if time_start > new_time:
            new_time = time_start

        new_time = (datetime.combine(datetime.today(), new_time) + object_time_treatment).time()

        if not first_iteration:
            __, return_path = warehouse.compute_manhattan_distance_with_BFS(last_state, positions[0], True)

            for pos in return_path:

                time_delta = compute_travel_time(pos, df_path.iloc[-1]['position'])
                new_time = (datetime.combine(datetime.today(), new_time) + time_delta).time()
                df_path.loc[len(df_path)] = [new_time, "Return","", pos]
                if pos == return_path[-1]:
                    new_time = (datetime.combine(datetime.today(), new_time) + object_time_treatment).time()

        for pos in positions:
            if df_path.empty:
                df_path.loc[len(df_path)] = [new_time, task_type, task_id, pos]
            else:
                time_delta = compute_travel_time(pos, df_path.iloc[-1]['position'])
                new_time = (datetime.combine(datetime.today(), new_time) + time_delta).time()
                df_path.loc[len(df_path)] = [new_time, task_type, task_id, pos]

        first_iteration = False
        last_state = positions[-1]

    return df_path


def schedule(bellman_matrix: np.ndarray, position_dict: Dict[Tuple[int, int, int], int],
                     csv_file_name: str, warehouse: Warehouse3D, num_drones: int) -> Dict[str, pd.DataFrame]:
    """Prepare the schedule for all drones."""
    csv_file_name = f'TL_{warehouse.name}.csv'
    csv = prepare_csv(bellman_matrix, position_dict, csv_file_name, warehouse, num_drones)
    drone_schedules = {f'd{i}': create_drone_schedule(csv, i, warehouse) for i in range(1, num_drones + 1)}

    return drone_schedules


