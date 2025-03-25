###############  Import section ###################

from datetime import time, datetime
import json
import random
import os
from Planification.adjacency_matrix import *
from Warehouse.warehouse_builder import load_config_warehouse,build_warehouse
from Planification.task_list_generator import create_objects_in_warehouse, generate_task_list
from Planification.planification import schedule
from Evitement.avoidance import count_calculated_collisions, count_direct_collisions, compute_cost, filter_indirect_collisions, detect_near_misses
from Evitement.optimisation import find_optimal_solution
from Visualisation.animation import launch_visualisation_plotly
from Planification.planification import load_config_planning

###############  Parameters ###################

warehouse_name = "warehouse1"
planning_config_name = "planning_test_1"
n_objects = 10
n_tasks = 14
arrival_time_slots = [time(8,0,0)]
departure_time_slots = [time(14,0,0)]

with open('Evitement/config_parameters_recuit.json', 'r') as file:
    configs_param = json.load(file)

config_name = 'config-8' #Choose the parameters configuration to use for the Simulated Annealing
config = configs_param[config_name]

# Accéder aux paramètres
collision_penalty = config["collision_penalty"]
avoidance_penalty = config["avoidance_penalty"]
total_duration_penalty = config["total_duration_penalty"]
T_init = config["T_init"]
T_freeze = config["T_freeze"]
alpha_T = config["alpha_T"]
k_iter = config["k_iter"]

seed = 29

###############  Code principal ###################

# Set the seed for reproductibility
np.random.seed(seed)
random.seed(seed)

# Load configs
warehouses_config, category_mapping = load_config_warehouse()
planning_config, mapping_config = load_config_planning(planning_config_name)
print(planning_config)

# Build warehouse
warehouse_3d = build_warehouse(warehouse_name, warehouses_config)

objects = create_objects_in_warehouse(n_objects, warehouse_3d)

# Build the list of tasks to accomplish during the day.
task_list_path = generate_task_list(n_tasks, objects, arrival_time_slots, departure_time_slots, warehouse_3d, mapping_config)

#False by default. If True, will display the warehouse in a plot
warehouse_3d.display(True)
warehouse_3d.show_graph()

#Generates the adjacency matrix
final_adjacency_matrix, coordinate_to_index = main_adjacency(warehouse_3d, category_mapping)

#Save the adjacency matrix generated for the warehouse in the folder AMatrix as a csv file
save_adj_matrix(final_adjacency_matrix, warehouse_3d.name)

#Call Bellman algorithm
# final_adjacency_matrix_2 = main_bellman(final_adjacency_matrix)

# Draw a first naive planning, that minimizes the total flight duration.
planning_drones = schedule(warehouse_3d, planning_config, mapping_config)

launch_visualisation_plotly(planning_drones,warehouse_3d)

print(planning_drones)

# Check if it respects the condition of no collisions and no near misses.
##Collision and near misses parameters
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
cost = compute_cost(planning_drones, planning_config['drone_speed'], charging_station_position, threshold, time_step, collision_penalty, avoidance_penalty, total_duration_penalty)
print("Cost:", cost)

experience = f"{planning_config['drone_quantity']}_drones&drone_speed={planning_config['drone_speed']}&collision_penalty={collision_penalty}&avoidance_penalty={avoidance_penalty}&T_init={T_init}&T_freeze={T_freeze}&alha_T={alpha_T}&k_iter={k_iter}"

results_dir = os.path.join("Results", warehouse_name, experience)
os.makedirs(results_dir, exist_ok=True)

# Use simulated annealing to find a solution that optimizes total flight duration while respecting the conditions.
final_planning, final_cost, respect_constraints = find_optimal_solution(results_dir, planning_drones, planning_config['drone_speed'], charging_station_position, threshold, time_step, collision_penalty, avoidance_penalty, total_duration_penalty, T_init, T_freeze, alpha_T, k_iter, 1)

print("Final planning : ", final_planning)
print("Final cost : ", final_cost)
print("Respect constraints: ", respect_constraints)

if not respect_constraints :
    direct_collisions_df, calculated_collisions_df = count_direct_collisions(planning_drones, charging_station_position), count_calculated_collisions(planning_drones, planning_config['drone_speed'], charging_station_position, time_step)
    calculated_collisions_df = filter_indirect_collisions(calculated_collisions_df, direct_collisions_df, time_step)
    print("Direct collisions: ", direct_collisions_df)
    print("Calculated collision: ", calculated_collisions_df)

    detect_near_misses_df = detect_near_misses(planning_drones, planning_config['drone_speed'], charging_station_position, threshold, time_step)
    print("Near misses: ", detect_near_misses_df)