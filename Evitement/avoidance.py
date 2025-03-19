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
        
            t1 = datetime.combine(datetime.today(), row1['time'].time())
            t2 = datetime.combine(datetime.today(), row2['time'].time())
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

                # 3 - Interpolate drone position at every minute only when suspect time interval and trajectories cross
                interpolated_points = interpolate_positions(p1, p2, q1, q2)

                for time, start_time1, start_time2, pos1, pos2 in interpolated_points:
                    #Check for crossing trajectories
                    prev_index = interpolated_points.index((time, start_time1, start_time2, pos1, pos2)) - 1
                    if prev_index >= 0:
                        _, prev_time1, prev_time2, prev_pos1, prev_pos2 = interpolated_points[prev_index]
                        if prev_pos1 == pos2 and prev_pos2 == pos1:
                            calculated_collisions.append((prev_time1, prev_time2, time, prev_pos1, prev_pos2, d1, d2))

    calculated_collisions_df = pd.DataFrame(calculated_collisions, columns=["start_time1", "start_time2", "collision_time", "pos1", "pos2", "drone1", "drone2"])
    
    return calculated_collisions_df

def detect_near_misses(drone_data, threshold=2):
    """Detects when drones are dangerously close accordingly to our threshold."""
    all_segments = {drone: get_segments(df) for drone, df in drone_data.items()}
    near_misses = []

    for (drone1, segments1), (drone2, segments2) in itertools.combinations(all_segments.items(), 2):
        for seg1 in segments1:
            for seg2 in segments2:
                interpolated_positions = interpolate_positions(*seg1, *seg2)

                for time, _, _, pos1, pos2 in interpolated_positions:
                    distance = sum(abs(a - b) for a, b in zip(pos1, pos2))

                    if distance <= threshold:
                        near_misses.append((time, drone1, drone2, pos1, pos2, distance))

    near_misses_df = pd.DataFrame(near_misses, columns=["time", "drone1", "drone2", "pos1", "pos2", "distance"])

    return near_misses_df

def compute_cost(drone_data: Dict[str, pd.DataFrame], threshold: int, collision_penalty: float = 100.0, avoidance_penalty: float = 10.0, total_duration_penalty: float = 1.0) -> float:
    """Compute cost of the total time of flight time, add a penalty weighted by the number of collision"""
    total_flight_time = 0
    start_times = []
    end_times = []

    for df in drone_data.values():

        total_time = 0
        last_valid_time = None  # Last time not charging
        in_recharge = False
        
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')

        start_times.append(df.iloc[0]["time"])  # First task start time
        end_times.append(df.iloc[-1]["time"])   # Last task end time

        for i, row in df.iterrows():
            if row['task_type'] == 'RC':  
                in_recharge = True  # Start charging, we stop the count
            elif in_recharge and row['task_type'] != 'RC':  
                # Drone leaves the charging station, counter starts back
                in_recharge = False
                last_valid_time = row['time']
            elif not in_recharge and last_valid_time is not None:
                # Sum up time actually flying
                total_time += (row['time'] - last_valid_time).total_seconds() / 60
                last_valid_time = row['time']

        total_flight_time += float(total_time)
    
    # Gets collisions
    direct_collisions_df, crossing_collisions_df = count_direct_collisions(drone_data), count_calculated_collisions(drone_data)
    total_collisions = len(direct_collisions_df) + len(crossing_collisions_df)

    # Gets near misses
    near_misses = detect_near_misses(drone_data, threshold)
    number_near_misses = len(near_misses)

    # Gets total duration to complete today's tasks
    start_times = [datetime.strptime(str(t), "%H:%M:%S") if isinstance(t, str) else t for t in start_times]
    end_times = [datetime.strptime(str(t), "%H:%M:%S") if isinstance(t, str) else t for t in end_times]
    total_duration = (max(end_times) - min(start_times)).total_seconds()

    return total_flight_time + (total_collisions * collision_penalty) + (number_near_misses * avoidance_penalty) + (total_duration * total_duration_penalty)

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