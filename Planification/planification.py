import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from warehouse import Warehouse3D
import itertools
from shapely.geometry import LineString
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

    csv_file['task_distance'] = csv_file.apply(
        lambda row: compute_distance(bellman_matrix, position_dict, row['pos0'],row['pos1']), axis=1)

    csv_file[['task_distance_2', 'task_path']] = csv_file.apply(
        lambda row: pd.Series(warehouse.compute_manhattan_distance_with_BFS(
            tuple(json.loads(row['pos0'])),
            tuple(json.loads(row['pos1'])),
            return_path=True,
            reduced=True
        )),
        axis=1
    )

    csv_file['task_time'] = csv_file['task_distance'] / drone_speed
    csv_file['task_time'] = csv_file['task_time'].apply(lambda x: timedelta(minutes=x))

    csv_file['task_time_2'] = csv_file['task_distance_2'] / drone_speed
    csv_file['task_time_2'] = csv_file['task_time_2'].apply(lambda x: timedelta(minutes=x))


    return csv_file


def create_drone_schedule(csv: pd.DataFrame, drone_number: int, warehouse: Warehouse3D, drone_speed : int) -> pd.DataFrame:
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
            __, return_path = warehouse.compute_manhattan_distance_with_BFS(last_state, positions[0], True, True)

            for pos in return_path:

                time_delta = compute_travel_time(pos, df_path.iloc[-1]['position'], drone_speed)
                new_time = (datetime.combine(datetime.today(), new_time) + time_delta).time()
                df_path.loc[len(df_path)] = [new_time, "Return","", pos]
                if pos == return_path[-1]:
                    new_time = (datetime.combine(datetime.today(), new_time) + object_time_treatment).time()

        for pos in positions:
            if df_path.empty:
                df_path.loc[len(df_path)] = [new_time, task_type, task_id, pos]
            else:
                time_delta = compute_travel_time(pos, df_path.iloc[-1]['position'], drone_speed)
                new_time = (datetime.combine(datetime.today(), new_time) + time_delta).time()
                df_path.loc[len(df_path)] = [new_time, task_type, task_id, pos]

        first_iteration = False
        last_state = positions[-1]

    return df_path


def schedule(bellman_matrix: np.ndarray, position_dict: Dict[Tuple[int, int, int], int],
                      warehouse: Warehouse3D, num_drones: int, drone_speed :int) -> Dict[str, pd.DataFrame]:
    """Prepare the schedule for all drones."""
    csv_file_name = f'TL_{warehouse.name}.csv'
    csv = prepare_csv(bellman_matrix, position_dict, csv_file_name, warehouse, num_drones, drone_speed)
    drone_schedules = {f'd{i}': create_drone_schedule(csv, i, warehouse, drone_speed) for i in range(1, num_drones + 1)}

    return drone_schedules

# Generates trajectories segments
def get_segments(df):
    segments = []
    for i in range(len(df) - 1):
        row1, row2 = df.iloc[i], df.iloc[i + 1]
        
        # Vérification du type de la colonne position
        if isinstance(row1['position'], str):
            x1, y1, z1 = eval(row1['position'])
            x2, y2, z2 = eval(row2['position'])
        else:
            x1, y1, z1 = row1['position']
            x2, y2, z2 = row2['position']
        
            t1 = datetime.combine(datetime.today(), row1['time'])
            t2 = datetime.combine(datetime.today(), row2['time'])
        segments.append(((x1, y1, t1), (x2, y2, t2)))  

    return segments

def interpolate_positions(p1, p2, q1, q2):
    x1, y1, t1 = p1
    x2, y2, t2 = p2
    x3, y3, t3 = q1
    x4, y4, t4 = q2

    common_start = max(t1, t3)
    common_end = min(t2, t4)

    if common_start >= common_end:
        return []

    # Generates interpolate positions when common time interval
    interpolated_positions = []
    duration = int((common_end - common_start).total_seconds() // 60)

    for step in range(duration + 1):
        time = common_start + pd.Timedelta(minutes=step)
        pos1_x = round(x1 + (x2 - x1) * (time - t1).total_seconds() / (t2 - t1).total_seconds())
        pos1_y = round(y1 + (y2 - y1) * (time - t1).total_seconds() / (t2 - t1).total_seconds())

        pos2_x = round(x3 + (x4 - x3) * (time - t3).total_seconds() / (t4 - t3).total_seconds())
        pos2_y = round(y3 + (y4 - y3) * (time - t3).total_seconds() / (t4 - t3).total_seconds())

        interpolated_positions.append((time, (pos1_x, pos1_y), (pos2_x, pos2_y)))

    return interpolated_positions

def count_direct_collisions(drone_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Counts direct collisions : same position at the same time"""

    all_positions = []
    for drone, df in drone_data.items():
        df_copy = df.copy()
        df_copy["drone"] = drone
        all_positions.append(df_copy)

    all_positions_df = pd.concat(all_positions, ignore_index=True)

    collisions_df = (
        all_positions_df
        .groupby(["time", "position"])
        .agg(drone_count=("drone", "count"), drones=("drone", lambda x: list(x)))
        .reset_index()
    )

    direct_collisions_df = collisions_df[collisions_df["drone_count"] > 1]
    return direct_collisions_df

def count_crossing_collisions(drone_data: Dict[str, pd.DataFrame]) -> pd.DataFrame :
    """Count crossing collisions between drones"""
    crossing_collisions = []

    # Gets segments for every drone trajectory
    drone_segments = {d: get_segments(df) for d, df in drone_data.items()}

    # Compare each unique pair of drones
    for d1, d2 in itertools.combinations(drone_segments.keys(), 2):
        segments1, segments2 = drone_segments[d1], drone_segments[d2]

        for (p1, p2) in segments1:
            for (q1, q2) in segments2:
                x1, y1, t1 = p1
                x2, y2, t2 = p2
                x3, y3, t3 = q1
                x4, y4, t4 = q2

                # 1 - Check for same time interval
                if max(t1, t3) > min(t2, t4):
                    continue  # No time intersection

                # 2 - Check for trajectories intersection
                traj1 = LineString([(x1, y1), (x2, y2)])
                traj2 = LineString([(x3, y3), (x4, y4)])

                if not traj1.intersects(traj2):
                    continue  # No spatial intersection trajectories

                # 3 - Interpolate drone position at every minute only when suspect time interval
                interpolated_points = interpolate_positions(p1, p2, q1, q2)

                for time, pos1, pos2 in interpolated_points:
                    #Check for crossing trajectories
                    prev_index = interpolated_points.index((time, pos1, pos2)) - 1
                    if prev_index >= 0:
                        prev_time, prev_pos1, prev_pos2 = interpolated_points[prev_index]
                        if prev_pos1 == pos2 and prev_pos2 == pos1:
                            crossing_collisions.append((prev_time, time, prev_pos1, prev_pos2, d1, d2))

    crossing_collisions_df = pd.DataFrame(crossing_collisions, columns=["start_time", "end_time", "pos1", "pos2", "drone1", "drone2"])
    
    return crossing_collisions_df

def count_collisions(drone_data): 
    return count_direct_collisions(drone_data), count_crossing_collisions(drone_data)

def compute_cost(drone_data: Dict[str, pd.DataFrame], collision_penalty: float = 10.0) -> float:
    """Compute cost of the total time of flight time, add a penalty weighted by the number of collision"""
    total_flight_time = 0

    for df in drone_data.values():
        if len(df) > 1:
            start_time = datetime.combine(datetime.today(), df.iloc[0]['time'])
            end_time = datetime.combine(datetime.today(), df.iloc[-1]['time'])
            total_flight_time += (end_time - start_time).total_seconds() / 60  # Time in minutes

    # Gets collisions
    direct_collisions_df, crossing_collisions_df = count_collisions(drone_data)
    total_collisions = len(direct_collisions_df) + len(crossing_collisions_df)

    return total_flight_time + (total_collisions * collision_penalty)