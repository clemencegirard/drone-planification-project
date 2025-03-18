from datetime import datetime, timedelta
import pandas as pd
from typing import Dict
import itertools
from shapely.geometry import LineString
import random

# Generates trajectories segments
def get_segments(df):
    segments = []
    for i in range(len(df) - 1):
        row1, row2 = df.iloc[i], df.iloc[i + 1]
        
        # Check column type
        if isinstance(row1['position'], str):
            x1, y1, z1 = eval(row1['position'])
            x2, y2, z2 = eval(row2['position'])
        else:
            x1, y1, z1 = row1['position']
            x2, y2, z2 = row2['position']
        
            t1 = datetime.combine(datetime.today(), row1['time'])
            t2 = datetime.combine(datetime.today(), row2['time'])
        segments.append(((x1, y1, z1, t1), (x2, y2, z2, t2)))  

    return segments

def interpolate_positions(p1, p2, q1, q2):
    x1, y1, z1, t1 = p1
    x2, y2, z2, t2 = p2
    x3, y3, z3, t3 = q1
    x4, y4, z4, t4 = q2

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
        pos1_z = round(z1 + (z2 - z1) * (time - t1).total_seconds() / (t2 - t1).total_seconds())

        pos2_x = round(x3 + (x4 - x3) * (time - t3).total_seconds() / (t4 - t3).total_seconds())
        pos2_y = round(y3 + (y4 - y3) * (time - t3).total_seconds() / (t4 - t3).total_seconds())
        pos2_z = round(z3 + (z4 - z3) * (time - t3).total_seconds() / (t4 - t3).total_seconds())

        interpolated_positions.append((time, t1, t3, (pos1_x, pos1_y, pos1_z), (pos2_x, pos2_y, pos2_z)))

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

def count_calculated_collisions(drone_data: Dict[str, pd.DataFrame]) -> pd.DataFrame :
    """Count calculated collisions between drones"""
    calculated_collisions = []

    # Gets segments for every drone trajectory
    drone_segments = {d: get_segments(df) for d, df in drone_data.items()}

    # Compare each unique pair of drones
    for d1, d2 in itertools.combinations(drone_segments.keys(), 2):
        segments1, segments2 = drone_segments[d1], drone_segments[d2]

        for (p1, p2) in segments1:
            for (q1, q2) in segments2:
                x1, y1, z1, t1 = p1
                x2, y2, z2, t2 = p2
                x3, y3, z3, t3 = q1
                x4, y4, z4, t4 = q2

                # 1 - Check for same time interval
                if max(t1, t3) > min(t2, t4):
                    continue  # No time intersection

                # 2 - Check for trajectories intersection
                traj1 = LineString([(x1, y1, z1), (x2, y2, z2)])
                traj2 = LineString([(x3, y3, z3), (x4, y4, z4)])

                if not traj1.intersects(traj2):
                    continue  # No spatial intersection trajectories

                # 3 - Interpolate drone position at every minute only when suspect time interval
                interpolated_points = interpolate_positions(p1, p2, q1, q2)

                for time, start_time1, start_time2, pos1, pos2 in interpolated_points:
                    #Check for crossing trajectories
                    prev_index = interpolated_points.index((time, start_time1, start_time2, pos1, pos2)) - 1
                    if prev_index >= 0:
                        _, prev_time1, prev_time2, prev_pos1, prev_pos2 = interpolated_points[prev_index]
                        if prev_pos1 == pos2 and prev_pos2 == pos1:
                            calculated_collisions.append((prev_time1, prev_time2, time, prev_pos1, prev_pos2, d1, d2))

    calculated_collisions_df = pd.DataFrame(calculated_collisions, columns=["start_time1", "start_time2", "end_time", "pos1", "pos2", "drone1", "drone2"])
    
    return calculated_collisions_df

def count_collisions(drone_data): 
    return count_direct_collisions(drone_data), count_calculated_collisions(drone_data)

def compute_cost(drone_data: Dict[str, pd.DataFrame], collision_penalty: float = 10.0) -> float:
    """Compute cost of the total time of flight time, add a penalty weighted by the number of collision"""
    total_flight_time = 0

    for df in drone_data.values():
    #     if len(df) > 1:
    #         start_time = datetime.combine(datetime.today(), df.iloc[0]['time'])
    #         end_time = datetime.combine(datetime.today(), df.iloc[-1]['time'])
    #         total_flight_time += (end_time - start_time).total_seconds() / 60  # Time in minutes

        total_time = 0
        last_valid_time = None  # Dernier instant hors recharge
        in_recharge = False  # Indicateur si le drone est en recharge
        
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')  # Conversion en datetime

        for i, row in df.iterrows():
            if row['task_type'] == 'RC':  
                in_recharge = True  # Début de recharge, on stoppe le comptage
            elif in_recharge and row['task_type'] != 'RC':  
                # Fin de recharge détectée, on reprend le comptage
                in_recharge = False
                last_valid_time = row['time']
            elif not in_recharge and last_valid_time is not None:
                # Accumulation du temps uniquement hors recharge
                total_time += (row['time'] - last_valid_time).total_seconds() / 60
                last_valid_time = row['time']

        total_flight_time += total_time
    
    # Gets collisions
    direct_collisions_df, crossing_collisions_df = count_collisions(drone_data)
    total_collisions = len(direct_collisions_df) + len(crossing_collisions_df)

    return total_flight_time + (total_collisions * collision_penalty)

def fix_collisions(planning: Dict[str, pd.DataFrame], direct_collisions: pd.DataFrame, calculated_collisions: pd.DataFrame):
    new_planning = planning.copy()

    # Fix direct collisions
    for _, collision in direct_collisions.iterrows():
        for drone in collision['drones'][1:]:
            # Change the time or the position to avoid the collision
            original_time = collision['time']
            new_time_delta = timedelta(minutes=random.uniform(-1, 1))
            new_time = (datetime.combine(datetime.today(), original_time) + new_time_delta).time()

            # Vérifiez que la condition correspond à au moins une ligne
            matching_indices = new_planning[drone]['time'] == original_time
            if matching_indices.any():
                new_planning[drone].loc[matching_indices, 'time'] = new_time

    # Fix crossing collisions
    for _, collision in calculated_collisions.iterrows():
        drone1 = collision['drone1']
        # Change the position of one of the drones to avoid the collision
        new_position1 = (collision['pos1'][0] + random.uniform(-1, 1),
                         collision['pos1'][1] + random.uniform(-1, 1))

        # Vérifiez que la condition correspond à au moins une ligne
        matching_indices = new_planning[drone1]['time'] == collision['start_time']
        if matching_indices.any():
            new_planning[drone1].loc[matching_indices, 'position'] = new_position1

    return new_planning

def make_new_planning(planning: Dict[str, pd.DataFrame]):
    new_planning = planning.copy()

    for drone, df in new_planning.items():
        for index, row in df.iterrows():
            if random.random() < 0.1:  # 10% de chance de modifier une ligne
                # Modifier l'heure de passage
                original_time = row['time']
                new_time_delta = timedelta(minutes=random.uniform(-1, 1))
                new_time = (datetime.combine(datetime.today(), original_time) + new_time_delta).time()
                new_planning[drone].loc[index, 'time'] = new_time

                # Modifier la position
                new_position = (row['position'][0] + random.uniform(-1, 1),
                                row['position'][1] + random.uniform(-1, 1),
                                row['position'][2] + random.uniform(-1, 1))
                new_planning[drone].loc[index, 'position'] = new_position

    return new_planning

def fix_calculated_collisions(planning: Dict[str, pd.DataFrame], calculated_collisions: pd.DataFrame, drone_speed : int):
    new_planning = planning.copy()

    # Fix calculated collisions
    for _, collision in calculated_collisions.iterrows():
        # The lane on which the collison will take place.
        if collision['pos1'][0] == collision['pos2'][0]:
            dimension_index = 0
        else :
            dimension_index = 1
        # Add new positions to one of the drones' trajectory to avoid the collision
        drone = collision['drone1']
        start_pos = collision['pos1']
        start_pos_time = collision['start_time1']

        # Bypass the collision
        new_planning[drone] = bypass_obstacle(new_planning[drone], start_pos, start_pos_time, dimension_index, drone_speed)

    return new_planning

def bypass_obstacle(planning: pd.DataFrame, start_pos: tuple[int, int, int], start_pos_time: str, dimension_index: int, drone_speed: int) :
    # Find the index of the start position
    start_pos_time = start_pos_time.time()
    start_pos_index = planning[planning['time'] == start_pos_time].index
    start_pos_index = int(start_pos_index[0])

    # Get the end position
    end_pos = planning.iloc[start_pos_index+1]['position']
    # Random offset to move on the next line.
    offset = random.choice([-1, 1])

    # Compute the new intermediate positions.
    if dimension_index == 0 :
        intermediate_pos_1 = (start_pos[0] + offset, start_pos[1], start_pos[2])
        intermediate_pos_2 = (end_pos[0] + offset, end_pos[1], end_pos[2])
    else :
        intermediate_pos_1 = (start_pos[0], start_pos[1] + offset, start_pos[2])
        intermediate_pos_2 = (end_pos[0], end_pos[1] + offset, end_pos[2])
    
    # Compute the new passage times.
    start_time = planning.iloc[start_pos_index]['time']
    end_time = planning.iloc[start_pos_index + 1]['time']
    
    intermediate_time_1 = (datetime.combine(datetime.today(), start_time) + timedelta(seconds=1 / drone_speed)).time()
    intermediate_time_2 = (datetime.combine(datetime.today(), end_time) + timedelta(seconds=1 / drone_speed)).time()

    # Remove microseconds
    intermediate_time_1 = intermediate_time_1.replace(microsecond=0)
    intermediate_time_2 = intermediate_time_2.replace(microsecond=0)

    # Retrieve other informations.
    task_type = planning.iloc[start_pos_index]['task_type']
    task_id = planning.iloc[start_pos_index]['id']

    # Build the new planning.
    new_rows = pd.DataFrame([
        {'time': intermediate_time_1, 'task_type': task_type, 'id': task_id, 'position': intermediate_pos_1},
        {'time': intermediate_time_2, 'task_type': task_type, 'id': task_id, 'position': intermediate_pos_2}
    ])

    new_planning = pd.concat([planning[:start_pos_index+1], new_rows, planning[start_pos_index+1:]]).reset_index(drop=True)

    # Delay passage time for next points.
    for i in range(start_pos_index + 3, len(new_planning)) :
        new_planning.at[i, 'time'] = (datetime.combine(datetime.today(), new_planning.at[i, 'time']) + timedelta(seconds=2 / drone_speed)).time().replace(microsecond=0)

    return new_planning