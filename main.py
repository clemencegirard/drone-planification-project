###############  Import section ###################

from datetime import time, datetime
import random
from Planification.adjacency_matrix import *
from Warehouse.warehouse_builder import load_config_warehouse,build_warehouse
from Planification.task_list_generator import create_objects_in_warehouse, generate_task_list
from Planification.planification import schedule
from Evitement.avoidance import count_calculated_collisions, count_direct_collisions, compute_cost, filter_indirect_collisions, detect_near_misses
from Evitement.optimisation import find_optimal_solution
from Visualisation.animation import launch_visualisation_plotly
from Planification.planification import load_config_planning

###############  Parameters ###################

warehouse_name = "intermediate_warehouse"
planning_config_name = "planning_test_1"
n_objects = 60
n_tasks = 120
arrival_time_slots = [time(8,0,0), time(10,0,0)]
departure_time_slots = [time(14,0,0), time(17,0,0)]

seed = 29

###############  Code principal ###################

# Set the seed for reproductibility
np.random.seed(seed)
random.seed(seed)

# Load configs
warehouses_config, category_mapping = load_config_warehouse()
planning_config = load_config_planning(planning_config_name)
print(planning_config)

# Build warehouse
warehouse_3d = build_warehouse(warehouse_name, warehouses_config)

objects = create_objects_in_warehouse(n_objects, warehouse_3d)

# Build the list of tasks to accomplish during the day.
task_list_path = generate_task_list(n_tasks, objects, arrival_time_slots, departure_time_slots, warehouse_3d)

#False by default. If True, will display the warehouse in a plot
warehouse_3d.display()
warehouse_3d.show_graph()

#Generates the adjacency matrix
final_adjacency_matrix, coordinate_to_index = main_adjacency(warehouse_3d, category_mapping)

#Save the adjacency matrix generated for the warehouse in the folder AMatrix as a csv file
save_adj_matrix(final_adjacency_matrix, warehouse_3d.name)

#Call Bellman algorithm
# final_adjacency_matrix_2 = main_bellman(final_adjacency_matrix)

# Draw a first naive planning, that minimizes the total flight duration.
planning_drones = schedule(warehouse_3d, planning_config)

launch_visualisation_plotly(planning_drones,warehouse_3d)

#print(planning_drones)

# Check if it respects the condition of no collisions and no near misses.
##Collision andnenar misses parameters
time_step = (60 // planning_config['drone_speed'])
threshold = 1 #security distance threshold. A value of 1 means that drones separated by 1 cell in the grid will be detected as near misses.
charging_station_position = tuple(warehouses_config[warehouse_name]['charging_station'][0]) #Filters out collisions happening on the charging station
##

direct_collisions_df, calculated_collisions_df = count_direct_collisions(planning_drones, charging_station_position), count_calculated_collisions(planning_drones, planning_config['drone_speed'], charging_station_position, time_step)
calculated_collisions_df = filter_indirect_collisions(calculated_collisions_df, direct_collisions_df, time_step)
print("Direct collisions: ", direct_collisions_df)
print("Calculated collision: ", calculated_collisions_df)

detect_near_misses_df = detect_near_misses(planning_drones, planning_config['drone_speed'], charging_station_position, threshold, time_step)
print("Near misses: ", detect_near_misses_df)

# Compute its cost.
cost = compute_cost(planning_drones, planning_config['drone_speed'], charging_station_position, threshold, time_step, collision_penalty = 500.0, avoidance_penalty= 10.0, total_duration_penalty = 1.0)
print("Cost:", cost)

# # # Use simulated annealing to find a solution that optimizes total flight duration while respecting the conditions.
# # final_planning, final_cost, respect_constraints = find_optimal_solution(planning_drones, 20, 0.1, 0.9, 20, 5)

# print("Final planning : ", final_planning)
# print("Final cost : ", final_cost)
# print("Respect constraints : ", respect_constraints)

# if not respect_constraints :
#     direct_collisions_df, calculated_collisions_df = count_collisions(final_planning)
#     print("Direct collisions:", direct_collisions_df)
#     print("Calculated collision: ", calculated_collisions_df)