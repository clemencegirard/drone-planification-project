###############  Import section ###################

from datetime import time, datetime
import json
import random
import os
from Planification.adjacency_matrix import *
from Warehouse.warehouse_builder import load_config_warehouse, build_warehouse
from Planification.task_list_generator import create_objects_in_warehouse, generate_task_list
from Planification.planification import schedule, load_config_planning
from Evitement.avoidance import (
    count_calculated_collisions, count_direct_collisions, compute_cost,
    filter_indirect_collisions, detect_near_misses
)
from Evitement.optimisation import find_optimal_solution
from Visualisation.animation import launch_visualisation_plotly
import copy
import pandas as pd

###############  Parameters ###################

# warehouse_name = "intermediate_warehouse_v2"
planning_config_name = "planning_boosted"
arrival_time_slots = [time(8, 0, 0)]
departure_time_slots = [time(14, 0, 0)]

seed = 29
verbose = True  # Mettre False pour désactiver les print

###############  Code principal ###################

# Set the seed for reproductibility
np.random.seed(seed)
random.seed(seed)

# Load configs
warehouses_config, category_mapping = load_config_warehouse()
planning_config, mapping_config = load_config_planning(planning_config_name)

# for warehouse_name in list(warehouses_config.keys()) :
for warehouse_name in ['warehouse1'] :

    if verbose:
        print(planning_config)

    # Build warehouse
    warehouse_3d = build_warehouse(warehouse_name, warehouses_config)
    objects = create_objects_in_warehouse(planning_config["n_objects"], warehouse_3d)

    # Build the list of tasks to accomplish during the day.
    task_list_path = generate_task_list(planning_config["n_tasks"], objects, arrival_time_slots, departure_time_slots, warehouse_3d,
                                        mapping_config)

    # Affichage optionnel du warehouse
    warehouse_3d.display(verbose)
    warehouse_3d.show_graph()

    # Génère la matrice d'adjacence
    final_adjacency_matrix, coordinate_to_index = main_adjacency(warehouse_3d, category_mapping)

    # Sauvegarde de la matrice d'adjacence
    save_adj_matrix(final_adjacency_matrix, warehouse_3d.name)

    # Génération du planning initial
    planning_drones = schedule(warehouse_3d, planning_config, mapping_config)

    launch_visualisation_plotly(planning_drones, warehouse_3d)

    # Vérification des collisions et near misses
    time_step = (60 // planning_config['drone_speed'])
    threshold = 1  # Distance minimale pour considérer un near miss
    charging_station_position = tuple(warehouses_config[warehouse_name]['charging_station'][0])

    direct_collisions_df = count_direct_collisions(planning_drones, charging_station_position)
    calculated_collisions_df = count_calculated_collisions(planning_drones, planning_config['drone_speed'],
                                                           charging_station_position, time_step)
    calculated_collisions_df = filter_indirect_collisions(calculated_collisions_df, direct_collisions_df, time_step)
    detect_near_misses_df = detect_near_misses(planning_drones, planning_config['drone_speed'], charging_station_position,
                                               threshold, time_step)

    if verbose:
        print("Direct collisions: ", direct_collisions_df)
        print("Calculated collision: ", calculated_collisions_df)
        print("Near misses: ", detect_near_misses_df)

    ###############  Début du recuit simulé ###################

    with open('Evitement/config_parameters_recuit.json', 'r') as file:
        configs_param_recuit = json.load(file)

    planning_drones_initial = copy.deepcopy(planning_drones)

    for config_name in configs_param_recuit:

        planning_drones = copy.deepcopy(planning_drones_initial)

        ################################ Configuration du recuit ##############################

        config_selected = configs_param_recuit[config_name]

        # Extraction des paramètres
        collision_penalty = config_selected["collision_penalty"]
        avoidance_penalty = config_selected["avoidance_penalty"]
        total_duration_penalty = config_selected["total_duration_penalty"]
        T_init = config_selected["T_init"]
        T_freeze = config_selected["T_freeze"]
        alpha_T = config_selected["alpha_T"]
        k_iter = config_selected["k_iter"]

        ################################ Initialisation #######################################

        # Calcul du coût initial
        cost = compute_cost(
            planning_drones, planning_config['drone_speed'], charging_station_position,
            threshold, time_step, collision_penalty, avoidance_penalty, total_duration_penalty
        )

        if verbose:
            print(f"Initial cost ({config_name}):", cost)

        experience = f"{planning_config['drone_quantity']}_drones&drone_speed={planning_config['drone_speed']}&collision_penalty={collision_penalty}&avoidance_penalty={avoidance_penalty}&T_init={T_init}&T_freeze={T_freeze}&alpha_T={alpha_T}&k_iter={k_iter}"

        results_dir = os.path.join("Results", planning_config_name, warehouse_name, experience)
        os.makedirs(results_dir, exist_ok=True)

        # Chemin du fichier où enregistrer les résultats
        results_file_path = os.path.join(results_dir, f"{config_name}_costs.txt")
        final_planning_file_path = os.path.join(results_dir, f"{config_name}_final_planning.json")

        # Sauvegarde des coûts initiaux
        with open(results_file_path, "w") as file:
            file.write(f"Configuration: {config_name}\n")
            file.write(f"Initial cost: {cost}\n")

        ################################ Algorithme de recuit simulé ##############################

        final_planning, final_cost, respect_constraints = find_optimal_solution(
            results_dir, planning_drones, planning_config['drone_speed'],
            charging_station_position, threshold, time_step,
            collision_penalty, avoidance_penalty, total_duration_penalty,
            T_init, T_freeze, alpha_T, k_iter, 1
        )

        launch_visualisation_plotly(final_planning, warehouse_3d)

        # Sauvegarde du coût final
        with open(results_file_path, "a") as file:  # "a" pour ajouter à la suite du fichier
            file.write(f"Final cost: {final_cost}\n")
            file.write(f"Respect constraints: {respect_constraints}\n")

        for key, df in final_planning.items():
            file_path = os.path.join(results_dir, f"{config_name}_final_planning_{key}.csv")
            df.to_csv(file_path, index=False)

        if verbose:
            print(f"Final planning ({config_name}): ", final_planning)
            print(f"Final cost ({config_name}): ", final_cost)
            print(f"Respect constraints ({config_name}): ", respect_constraints)

        if not respect_constraints and verbose:
            direct_collisions_df = count_direct_collisions(planning_drones, charging_station_position)
            calculated_collisions_df = count_calculated_collisions(planning_drones, planning_config['drone_speed'],
                                                                   charging_station_position, time_step)
            calculated_collisions_df = filter_indirect_collisions(calculated_collisions_df, direct_collisions_df, time_step)

            print(f"Direct collisions ({config_name}): ", direct_collisions_df)
            print(f"Calculated collision ({config_name}): ", calculated_collisions_df)

            detect_near_misses_df = detect_near_misses(planning_drones, planning_config['drone_speed'],
                                                       charging_station_position, threshold, time_step)
            print(f"Near misses ({config_name}): ", detect_near_misses_df)
