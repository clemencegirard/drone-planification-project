from datetime import datetime, timedelta
import pandas as pd
from typing import Dict
import itertools
from shapely.geometry import LineString
import random
import time

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

def interpolate_positions(p1, p2, q1, q2, drone_speed):
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
    time_step = (60//drone_speed)/3
    duration = int(((common_end - common_start).total_seconds())//time_step)

    for step in range(duration + 1):
        time = common_start + pd.Timedelta(seconds=step * time_step)
        pos1_x = round(x1 + (x2 - x1) * (time - t1).total_seconds() / (t2 - t1).total_seconds())
        pos1_y = round(y1 + (y2 - y1) * (time - t1).total_seconds() / (t2 - t1).total_seconds())
        pos1_z = round(z1 + (z2 - z1) * (time - t1).total_seconds() / (t2 - t1).total_seconds())

        pos2_x = round(x3 + (x4 - x3) * (time - t3).total_seconds() / (t4 - t3).total_seconds())
        pos2_y = round(y3 + (y4 - y3) * (time - t3).total_seconds() / (t4 - t3).total_seconds())
        pos2_z = round(z3 + (z4 - z3) * (time - t3).total_seconds() / (t4 - t3).total_seconds())

        interpolated_positions.append((time, t1, t3, (pos1_x, pos1_y, pos1_z), (pos2_x, pos2_y, pos2_z)))

    return interpolated_positions

def count_direct_collisions(drone_data: Dict[str, pd.DataFrame], charging_station_position: tuple) -> pd.DataFrame:
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

    direct_collisions_df = collisions_df[
        (collisions_df["drone_count"] > 1) & 
        (collisions_df["drones"].apply(lambda x: len(set(x)) > 1)) &
        (collisions_df["position"]!= charging_station_position)] #there are no collisions on the charging station
    
    return direct_collisions_df

def filter_indirect_collisions(calculated_collisions_df, direct_collisions_df, time_step):
    """Remove calculated collisions that are too close to direct collisions."""
    direct_collisions_times = direct_collisions_df['time']

    mask = calculated_collisions_df.apply(
        lambda row: not any(
            abs((row['collision_time'] - dt).total_seconds()) < time_step or
            abs((row['start_time1'] - dt).total_seconds()) < time_step or
            abs((row['start_time2'] - dt).total_seconds()) < time_step
            for dt in direct_collisions_times
        ),
        axis=1
    )

    return calculated_collisions_df[mask]

def count_calculated_collisions(drone_data: Dict[str, pd.DataFrame], drone_speed: float, charging_station_position: tuple, time_step: float) -> pd.DataFrame :
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
                interpolated_points = interpolate_positions(p1, p2, q1, q2, drone_speed)

                for time, start_time1, start_time2, pos1, pos2 in interpolated_points:
                    #Check for crossing trajectories
                    prev_index = interpolated_points.index((time, start_time1, start_time2, pos1, pos2)) - 1
                    if prev_index >= 0:
                        _, prev_time1, prev_time2, prev_pos1, prev_pos2 = interpolated_points[prev_index]
                        if prev_pos1 == pos2 and prev_pos2 == pos1:
                            calculated_collisions.append((prev_time1, prev_time2, time, prev_pos1, d1, d2))

    calculated_collisions_df = pd.DataFrame(calculated_collisions, columns=["start_time1", "start_time2", "collision_time", "collision_position", "drone1", "drone2"])

    #Filter collision happening within the same step to avoid duplicates
    calculated_collisions_df["time_diff"] = (
    calculated_collisions_df.groupby(["drone1", "drone2", "collision_position"])["collision_time"]
    .diff()
    .dt.total_seconds()
    )
    calculated_collisions_df = calculated_collisions_df[
        (calculated_collisions_df["time_diff"].isna()) | (calculated_collisions_df["time_diff"] >= time_step)
    ].drop(columns=["time_diff"])

    #Remove collisions if happening on the charging station
    calculated_collisions_df = calculated_collisions_df[
    ~(calculated_collisions_df["collision_position"] == charging_station_position)]

    return calculated_collisions_df

def detect_near_misses(drone_data, drone_speed, charging_station_position, threshold, time_step):
    """Detects when drones are dangerously close accordingly to our threshold."""
    all_segments = {drone: get_segments(df) for drone, df in drone_data.items()}
    near_misses = []

    for (drone1, segments1), (drone2, segments2) in itertools.combinations(all_segments.items(), 2):
        for seg1 in segments1:
            for seg2 in segments2:
                interpolated_positions = interpolate_positions(*seg1, *seg2, drone_speed)

                for time, _, _, pos1, pos2 in interpolated_positions:
                    distance = sum(abs(a - b) for a, b in zip(pos1, pos2))

                    if distance <= threshold and distance > 0 :
                        near_misses.append((time, drone1, drone2, pos1, pos2))

    near_misses_df = pd.DataFrame(near_misses, columns=["time", "drone1", "drone2", "pos1", "pos2"])
    
    #Remove near misses happening at the charging station
    near_misses_df = near_misses_df = near_misses_df[
    ~((near_misses_df["pos1"] == charging_station_position) & 
      (near_misses_df["pos2"] == charging_station_position))]
    
    #Filter near misses happening within the same step to avoid duplicates
    near_misses_df["time_diff"] = (
        near_misses_df.groupby(["drone1", "drone2", "pos1", "pos2"])["time"]
        .diff()
        .dt.total_seconds()
    )
    near_misses_df = near_misses_df[
        near_misses_df["time_diff"].isna() | (near_misses_df["time_diff"] > time_step)
    ]
    near_misses_df = near_misses_df.drop(columns=["time_diff"])

    return near_misses_df

def compute_cost(drone_data: Dict[str, pd.DataFrame], drone_speed: int, charging_station_position: tuple,
                 threshold: int, time_step: float, collision_penalty: float = 100.0, avoidance_penalty: float = 10.0,
                 total_duration_penalty: float = 1.0) -> float:
    """Compute cost of the total time of flight time, add a penalty weighted by the number of collision"""

    # start_time = time.time()  # Enregistre l'heure de début

    total_duration = 0
    total_recharge_time = 0

    # Get total flight duration for every drone and sum up
    for drone, df in drone_data.items():
        df["time"] = pd.to_datetime(df["time"])
        min_time = df["time"].min()
        max_time = df["time"].max()
        duration = (max_time - min_time).total_seconds()
        total_duration += duration

        df["next_task"] = df["task_type"].shift(-1)
        df["next_time"] = df["time"].shift(-1)

        recharge_intervals = df[(df["task_type"] == "RC") & (df["next_task"] == "PO")]

        # Force datetime type for subtraction
        recharge_intervals.loc[:, "next_time"] = pd.to_datetime(recharge_intervals["next_time"], errors='coerce')
        recharge_intervals.loc[:, "time"] = pd.to_datetime(recharge_intervals["time"], errors='coerce')

        recharge_time = (recharge_intervals["next_time"] - recharge_intervals["time"]).sum().total_seconds()
        total_recharge_time += recharge_time

    # Gets collisions
    direct_collisions_df, crossing_collisions_df = count_direct_collisions(drone_data,
                                                                           charging_station_position), count_calculated_collisions(
        drone_data, drone_speed, charging_station_position, time_step)
    total_collisions = len(direct_collisions_df) + len(crossing_collisions_df)

    # Gets near misses
    near_misses = detect_near_misses(drone_data, drone_speed, charging_station_position, threshold, time_step)
    number_near_misses = len(near_misses)

    total_flight_time = total_duration - total_recharge_time



    return total_flight_time + (total_collisions * collision_penalty) + (number_near_misses * avoidance_penalty) + (
                total_duration * total_duration_penalty)

# Change the drone planning to another close solution, using the given heuristic.
def change_planning(planning: Dict[str, pd.DataFrame],heuristic: int, direct_collisions: pd.DataFrame, calculated_collisions: pd.DataFrame, near_misses: pd.DataFrame):
    if heuristic == 1 :
        return fix_direct_collisions_time(planning, direct_collisions)
    elif heuristic == 2 :
        return fix_calulated_collisions_time(planning, calculated_collisions)
    elif heuristic == 3 :
        return fix_near_misses_time(planning, near_misses)
    
    return planning

# Create a new solution for the simulated annealing.
def make_new_planning(planning: Dict[str, pd.DataFrame], drone_speed: int, charging_station_position: tuple, threshold: int, time_step: float):
    # Identify collisions and near misses that make the solution not acceptable.
    direct_collisions = count_direct_collisions(planning, charging_station_position)
    calculated_collisions = count_calculated_collisions(planning, drone_speed, charging_station_position, time_step)
    near_misses = detect_near_misses(planning, drone_speed, charging_station_position, threshold, time_step)

    possible_heuristics = []
    if not direct_collisions.empty :
        possible_heuristics.append(1)
    elif not calculated_collisions.empty :
        possible_heuristics.append(2)
    elif not near_misses.empty :
        possible_heuristics.append(3)
    
    if len(possible_heuristics) != 0 :
        heuristic = random.choice(possible_heuristics)
    else : heuristic = 0

    new_planning = change_planning(planning, heuristic, direct_collisions, calculated_collisions, near_misses)

    return new_planning

# Fix a calculated collision by modifying one of the drone's trajectory.
# NOT USED FOR NOW.
def fix_calculated_collisions_bypass(planning: Dict[str, pd.DataFrame], calculated_collisions: pd.DataFrame, drone_speed : int):
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

# Change a drone's trajectory to bypass an obstacle.
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

# Retrieve a drone's level of battery at a given time.
def get_battery_at_time(planning: Dict[str, pd.DataFrame], drone: str, collision_time: str) -> float:
    df = planning[drone].copy()
    df['time'] = pd.to_datetime(df['time'])
    collision_time = pd.Timestamp(collision_time)
 
    closest_row = df.loc[(df['time'] - collision_time).abs().idxmin()]
    return closest_row['battery_percentage']

# Fix a direct collision by delaying one of the drones involved.
def fix_direct_collisions_time(planning_drone: Dict[str, pd.DataFrame], collisions: pd.DataFrame):
    # Create a copy of the planning to change
    new_planning = planning_drone.copy()
    # Fix one collision of the list
    collision_to_fix_index = random.choice([i for i in range(0, len(collisions))])
    collision = collisions.iloc[collision_to_fix_index]
    collision_time = collision['time']
    drones = collision['drones']

    # Select drones according to battery level at the time of the collision
    battery_levels = {drone: get_battery_at_time(new_planning, drone, collision_time) for drone in drones}
    sorted_drones = sorted(battery_levels, key=battery_levels.get)
    # If there are two drones, delay the one with the lowest battery
    if len(drones) == 2 :
        drone = sorted_drones[0]
        planning_drone = new_planning[drone]
        new_planning[drone] = push_back_transit_times(planning_drone, collision_time, time_offset=2)
    # Else, choose drones to delay
    else :
        # Fix one drone
        fixed_drone = sorted_drones[len(sorted_drones) // 2]
        for drone in sorted_drones:
            if drone == fixed_drone:
                continue 
            # Delay the drone with the lowest battery
            if drone == sorted_drones[0]:  
                time_offset = random.choice([1, 2])  
            # Advance drones with highest battery
            else:  
                time_offset = random.choice([-2, -1])
            planning_drone = new_planning[drone]
            new_planning[drone] = push_back_transit_times(planning_drone, collision_time, time_offset)
    
    return new_planning

# Fix a calculated collision by delaying one of the drones involved.
def fix_calulated_collisions_time(planning_drone: Dict[str, pd.DataFrame], collisions: pd.DataFrame):
    # Create a copy of the planning to change
    new_planning = planning_drone.copy()
    # Fix one collision of the list
    collision_to_fix_index = random.choice([i for i in range(0, len(collisions))])
    collision = collisions.iloc[collision_to_fix_index]
    collision_time = collision['collision_time']
    drone1, drone2 = collision['drone1'], collision['drone2']

    # Select drones according to battery level at the time of the collision
    battery_levels = {drone: get_battery_at_time(new_planning, drone, collision_time) for drone in [drone1, drone2]}
    sorted_drones = sorted(battery_levels, key=battery_levels.get)
    # Delay the drone with the lowest battery
    drone = sorted_drones[0]
    planning_drone = new_planning[drone]
    new_planning[drone] = push_back_transit_times(planning_drone, collision_time, time_offset=2)
    
    return new_planning

# Fix a near miss by delaying one of the drones involved.
def fix_near_misses_time(planning_drone: Dict[str, pd.DataFrame], near_misses: pd.DataFrame):
    # Create a copy of the planning to change
    new_planning = planning_drone.copy()

    # Fix one collision of the list
    near_miss_to_fix_index = random.choice([i for i in range(0, len(near_misses))])
    near_miss = near_misses.iloc[near_miss_to_fix_index]
    near_miss_time = near_miss['time']
    drone1, drone2 = near_miss['drone1'], near_miss['drone2']

    # Select drones according to battery level at the time of the near miss
    battery_levels = {drone: get_battery_at_time(new_planning, drone, near_miss_time) for drone in [drone1, drone2]}
    sorted_drones = sorted(battery_levels, key=battery_levels.get)

    # Delay the drone with the lowest battery
    drone = sorted_drones[0]
    planning_drone = new_planning[drone]
    new_planning[drone] = push_back_transit_times(planning_drone, near_miss_time, time_offset=2)

    return new_planning

# Delay a given drone's planning for one task, between two visits at the charging station.
def push_back_transit_times(planning: pd.DataFrame, collision_time: pd.Timestamp, time_offset: int):
    # Find the index of the collision
    collision_time_index = planning.index[planning['time'] == collision_time].tolist()
    
    # If there is no row matching the exact collision time, use the one just before
    if not collision_time_index:
        closest_time_index = planning[planning['time'] < collision_time]['time'].idxmax()
        collision_time_index = [closest_time_index]  # Keep the structure
        
    collision_time_index = collision_time_index[0]  # Get the first match

    # Find the last time the drone was at the charging station before the collision
    previous_charging_index = None
    for i in range(collision_time_index, -1, -1):  # Iterate backwards
        if planning.at[i, 'task_type'] == 'RC':
            previous_charging_index = i
            break

    # If the drone never visited the station before, it is because it's its first trajectory.
    if previous_charging_index is None:
        previous_charging_index = 0

    # Find the next time the drone will be at the charging station after the collision
    next_charging_index = None
    for i in range(collision_time_index, len(planning)) :
        if planning.at[i, 'task_type'] == 'RC':
            next_charging_index = i
            break

    # If the drone never visits the station after, it's because it's its last trajectory.
    if next_charging_index is None:
        last_charging_index = len(planning) 
    # Find the last index at the charging station for the drone
    else :
        for i in range(next_charging_index, len(planning)):
            if planning.at[i, 'task_type'] == 'RC':
                last_charging_index = i
            else:
                break

    # Delay passage time between the two RC times.
    for i in range(previous_charging_index, last_charging_index+1):
        new_time = planning.at[i,'time'] + pd.Timedelta(minutes=time_offset)

        planning.at[i, 'time'] = new_time
    
    # Adjust subsequent tasks if needed to avoid overlaps
    i = last_charging_index
    while i + 1 < len(planning) and planning.at[i, 'time'] >= planning.at[i + 1, 'time']:
        planning.at[i + 1, 'time'] += pd.Timedelta(minutes=time_offset)
        i += 1

    return planning