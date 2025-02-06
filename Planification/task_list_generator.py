import csv
import random
import string
from pathlib import Path
from warehouse import Warehouse3D, Object
from warehouse_builder import *

def generate_object_id(length=8):
    """Generate a random alphanumeric string of uppercase letters and digits."""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def create_objects_in_warehouse(n: int, warehouse: Warehouse3D):
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

    for _ in range(n):
        object_id = generate_object_id()
        is_on_shelf = bool(random.randint(0, 1))

        # Only attempt placement if there are empty slots.
        if is_on_shelf and available_positions:
            # Choose a random available spot
            row, col, height = random.choice(available_positions)
            warehouse.add_object(row, col, height)
            available_positions.remove((row, col, height))
        else:
            # If there are no more available positions, the object will arrive on the arrival slots.
            is_on_shelf = False
            row, col, height = random.choice(arrival_slot_positions)

        object = Object(object_id, is_on_shelf, row, col, height)
        objects.append(object)

    return objects
 
def generate_task_list(n: int, objects: list[Object], warehouse: Warehouse3D):
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
    departure_slot_positions = [
        (r, c, h) for r in range(warehouse.rows)
        for c in range(warehouse.cols)
        for h in range(warehouse.height)
        if warehouse.mat[r, c, h] == 6
    ]

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header.
        writer.writerow(["id", "row0", "col0", "height0", "row1", "col1", "height1"])
        
        for i in range(n):
            # Prevent empty list errors.
            if not objects_movable:
                logging.info(f"Warning: No more movable objects available before having listed all {n} tasks. There will be {i} tasks.")
                break
            object = random.choice(objects_movable)
            initial_row = object.row
            initial_col = object.col
            initial_height = object.height
            # If the object is on a shelf, it can only be moved to the departure slots.
            # Else, it means it is at the arrival slots, and it has to be moved to a shelf in the warehouse.
            if object.is_on_shelf :
                final_row, final_col, final_height = random.choice(departure_slot_positions)
                objects_movable.remove(object)
            else :
                final_row, final_col, final_height = random.choice(shelves_positions)
                object.is_on_shelf = True
                object.move_to(final_row, final_col, final_height)

            row = [
                object.id,
                initial_row,
                initial_col,
                initial_height,
                final_row,
                final_col,
                final_height,
            ]
            writer.writerow(row)
    
    print(f"File '{file_name}' generated with {n} rows.")