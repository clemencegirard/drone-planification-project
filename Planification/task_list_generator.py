import csv
import random
import string
import copy
from datetime import time
from pathlib import Path
from warehouse import Warehouse3D, Object
from warehouse_builder import *

def generate_object_id(length=8):
    """Generate a random alphanumeric string of uppercase letters and digits."""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def create_objects_in_warehouse(n_objects: int, warehouse: Warehouse3D):
    objects = []
    # Available shelf positions in the warehouse.
    available_positions = [
        (r, c, h) for r in range(warehouse.rows)
        for c in range(warehouse.cols)
        for h in range(warehouse.height)
        if warehouse.mat[r, c, h] == 2
    ]
    # Potential positions of objects on arrival slots.
    arrival_slot_positions = [
        (r, c, h) for r in range(warehouse.rows)
        for c in range(warehouse.cols)
        for h in range(warehouse.height)
        if warehouse.mat[r, c, h] == 5
    ]

    for _ in range(n_objects):
        object_id = generate_object_id()
        is_on_shelf = bool(random.randint(0, 1))

        # Only attempt placement if there are empty slots.
        if is_on_shelf and available_positions:
            # Choose a random available spot
            row, col, height = random.choice(available_positions)
            warehouse.add_object(row, col, height)
            available_positions.remove((row, col, height))
        elif arrival_slot_positions :
            # If there are no more available positions, the object will arrive on the arrival slots.
            is_on_shelf = False
            row, col, height = random.choice(arrival_slot_positions)
        else :
            logging.log(1, "No empty spot available and no arrival mat")
            break

        object = Object(object_id, is_on_shelf, row, col, height)
        objects.append(object)

    return objects
 
def choose_slot_and_time(times_positions: dict[time, dict[tuple[int, int, int], int]]) -> tuple[int, int, int]:
    # Iterate over the times in chronological order
    for departure_time in sorted(times_positions.keys()):
        slots = times_positions[departure_time]  # Get the slots for this time
        
        # Check each slot at this time
        for slot_pos, capacity in slots.items():
            if capacity > 0:  # If slot has space, assign object here
                slots[slot_pos] -= 1  # Decrease capacity
                return departure_time, slot_pos  # Return selected time and slot
            
    return None, None

def generate_task_list(n_tasks: int, objects: list[Object], arrival_times: list[time], departure_times: list[time], warehouse: Warehouse3D):
    """Generate a CSV file with n rows of the following format :

    object_id  row0 col0 height0  row1 col1 height1

    where the object has to go from position 0 to position 1"""

    current_path = Path(__file__).parent.resolve()
    task_list_dir = current_path / "TaskList"
    file_name = f'TL_{warehouse.name}.csv'
    file_path = task_list_dir / file_name

    objects_movable = objects[:]
    shelves_positions = [
        (r, c, h) for r in range(warehouse.rows)
        for c in range(warehouse.cols)
        for h in range(warehouse.height)
        if (warehouse.mat[r, c, h] == 2 or warehouse.mat[r, c, h] == 3)
    ]
    slots_capacity = warehouse.mat_capacity
    # Departure slots
    departure_slot_positions = [
        (r, c, h) for r in range(warehouse.rows)
        for c in range(warehouse.cols)
        for h in range(warehouse.height)
        if warehouse.mat[r, c, h] == 6
    ]
    departure_slots = {slot_position: slots_capacity for slot_position in departure_slot_positions}
    departures = {departure_time: copy.deepcopy(departure_slots) for departure_time in departure_times}
    # Arrival slots
    arrival_slot_positions = [
        (r, c, h) for r in range(warehouse.rows)
        for c in range(warehouse.cols)
        for h in range(warehouse.height)
        if warehouse.mat[r, c, h] == 5
    ]
    arrival_slots = {slot_position: slots_capacity for slot_position in arrival_slot_positions}
    arrivals = {arrival_time: copy.deepcopy(arrival_slots) for arrival_time in arrival_times}

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header.
        writer.writerow(["task_type", "id", "row0", "col0", "height0", "row1", "col1", "height1", "time"])
        
        for i in range(n_tasks):
            # Prevent empty list errors.
            if not objects_movable:
                logging.info(f"Warning: No more movable objects available before having listed all {n_tasks} tasks. There will be {i} tasks.")
                break
            object = random.choice(objects_movable)
            initial_row = object.row
            initial_col = object.col
            initial_height = object.height
            # If the object is on a shelf, it can only be moved to the departure slots.
            # Else, it means it is at the arrival slots, and it has to be moved to a shelf in the warehouse.
            if object.is_on_shelf :
                task_type = 'D'
                time, departure_pos = choose_slot_and_time(departures)
                if not time or not departure_pos :
                    logging.info("Warehouse can not ship anymore items for the day")
                    objects_movable = [object for object in objects_movable if not object.is_on_shelf]
                    continue
                else :
                    final_row, final_col, final_height = departure_pos
                    objects_movable.remove(object)
            else :
                task_type = 'A'
                time, arrival_pos = choose_slot_and_time(arrivals)
                if not time or not arrival_pos :
                    logging.info("Warehouse can receive no more items for the day")
                    objects_movable = [object for object in objects_movable if object.is_on_shelf]
                    continue
                else :
                    initial_row, initial_col, initial_height = arrival_pos
                    final_row, final_col, final_height = random.choice(shelves_positions)
                    object.is_on_shelf = True
                    object.move_to(final_row, final_col, final_height)

            row = [
                task_type,
                object.id,
                initial_row,
                initial_col,
                initial_height,
                final_row,
                final_col,
                final_height,
                time
            ]
            writer.writerow(row)
    
    print(f"File '{file_name}' generated with {n_tasks} rows.")

    return file_name
    