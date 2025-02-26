###############  Import section ###################

from datetime import time
import random
from adjacency_matrix import *
from warehouse_builder import load_config,build_warehouse
from bellman import main_bellman
from task_list_generator import create_objects_in_warehouse, generate_task_list
from planification import schedule, count_collisions, get_segments

###############  Parameters ###################

warehouse_name = "one_level_U_warehouse"
n_objects = 60
n_tasks = 120
arrival_time_slots = [time(8,0,0), time(10,0,0)]
departure_time_slots = [time(14,0,0), time(17,0,0)]

seed = 29

###############  Code principal ###################

# Set the seed for reproductibility
np.random.seed(seed)
random.seed(seed)

# Load config
warehouses_config, category_mapping = load_config()

# Build warehouse
warehouse_3d = build_warehouse(warehouse_name, warehouses_config)

objects = create_objects_in_warehouse(n_objects, warehouse_3d)
# objects =[]

# Build the list of tasks to accomplish during the day.
task_list_path = generate_task_list(n_tasks, objects, arrival_time_slots, departure_time_slots, warehouse_3d)

#False by default. If True, will display the warehouse in a plot
warehouse_3d.display(True)
warehouse_3d.show_graph()

#Generates the adjacency matrix
final_adjacency_matrix, coordinate_to_index = main_adjacency(warehouse_3d, category_mapping)
print(final_adjacency_matrix)
#Save the adjacency matrix generated for the warehouse in the folder AMatrix as a csv file
save_adj_matrix(final_adjacency_matrix, warehouse_name)

#Call Bellman algorithm
final_adjacency_matrix_2 = main_bellman(final_adjacency_matrix)

planning_drones = schedule(final_adjacency_matrix_2, coordinate_to_index, warehouse_name, warehouse_3d, 3)
print("d1:", planning_drones["d1"].iloc[10:30])
print("d2:", planning_drones["d2"].iloc[10:30])
print("d3:", planning_drones["d3"].iloc[10:30])


direct_collisions_df, crossing_collisions_df = count_collisions(planning_drones)
print("Direct collisions:", direct_collisions_df)
print("Potential collision: ", crossing_collisions_df)