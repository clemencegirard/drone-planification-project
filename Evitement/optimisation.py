import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import random
from tqdm import tqdm
from Evitement.avoidance import *

def plot_evolution(data, xlabel, ylabel, title, y_log=False, save_path=None):
    plt.figure()
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if y_log == True : plt.yscale('log')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    plt.show()

def metropolis_acceptance(current_cost, new_cost, temp):
    if new_cost < current_cost:
        return True
    return random.uniform(0, 1) < np.exp((current_cost - new_cost) / temp)

def simulated_annealing(results_dir: str, planning: Dict[str, pd.DataFrame], drone_speed: int, charging_station_position: tuple, threshold: int, time_step: float, collision_penalty: float, avoidance_penalty: float, total_duration_penalty: float, t_initial: float = 1000, t_freeze: float = 0.1, alpha: float = 0.9, iterations_per_temp = 500):

    # Intitialisation
    temp = t_initial
    current_planning = planning.copy()
    current_cost = compute_cost(current_planning, drone_speed, charging_station_position, threshold, time_step, collision_penalty, avoidance_penalty, total_duration_penalty)

    # Save data for visualisation
    costs_evol = [current_cost]
    temp_evol = []
    accept_evol = []

    # Keep the best solution
    best_planning = current_planning

    # Visualize the function's progression
    pbar = tqdm(total=int(np.log(t_freeze / t_initial) / np.log(alpha)), desc="Simulated Annealing")

    while temp > t_freeze :

        acceptances = 0
        # Explore solutions at constant temperature

        for _ in range(iterations_per_temp) :

            # Generate a slightly different new planning
            new_planning = make_new_planning(current_planning, drone_speed, charging_station_position, threshold, time_step)

            # Compare costs
            new_cost = compute_cost(new_planning, drone_speed, charging_station_position, threshold, time_step, collision_penalty, avoidance_penalty, total_duration_penalty)

            if metropolis_acceptance(current_cost, new_cost, temp) :
                current_planning, current_cost = new_planning, new_cost
                acceptances += 1
            
            if new_cost < current_cost :
                best_planning = new_planning

        # Keep trace
        costs_evol.append(current_cost)
        temp_evol.append(temp)
        accept_evol.append(acceptances/iterations_per_temp)

        # Cool the temperature
        temp = temp * alpha
        pbar.update(1)

    pbar.close()

    # Plot evolutions
    plot_evolution(costs_evol, "Iterations", "Cost", "Cost evolution", True, os.path.join(results_dir, "cost_evolutions"))
    plot_evolution(temp_evol, "Iterations", "Temperature", "Temperature evolution", False, os.path.join(results_dir, "temp_evolutions"))
    plot_evolution(accept_evol, "Steps", "Acceptation rate", "Acceptation rate evolution", True, os.path.join(results_dir, "acceptance_rate_evolutions"))

    return best_planning

# Recursively finds an optimal solution that respects the constraints.
def find_optimal_solution(results_dir: str, planning: Dict[str, pd.DataFrame], drone_speed: int, charging_station_position: tuple, threshold: int,time_step: float, collision_penalty: float, avoidance_penalty: float, total_duration_penalty: float, t_initial: float, t_freeze: float, alpha: float, iterations_per_temp, max_iterations: int):

    direct_collisons, calculated_collisions = count_direct_collisions(planning, charging_station_position), count_calculated_collisions(planning,  drone_speed, charging_station_position, time_step)
    calculated_collisions = filter_indirect_collisions(calculated_collisions, direct_collisons, time_step)
    near_misses = detect_near_misses(planning, drone_speed, charging_station_position, threshold, time_step)

    if direct_collisons.empty and calculated_collisions.empty and near_misses.empty :
        return planning, compute_cost(planning, drone_speed, charging_station_position, threshold, time_step, collision_penalty, avoidance_penalty, total_duration_penalty), True
    elif max_iterations == 0 :
        return planning, compute_cost(planning, drone_speed, charging_station_position, threshold, time_step, collision_penalty, avoidance_penalty, total_duration_penalty), False

    new_planning = simulated_annealing(results_dir, planning, drone_speed, charging_station_position, threshold, time_step, collision_penalty, avoidance_penalty, total_duration_penalty, t_initial, t_freeze, alpha, iterations_per_temp)
    max_iterations -= 1
    return find_optimal_solution(results_dir, new_planning, drone_speed, charging_station_position, threshold, time_step, collision_penalty, avoidance_penalty, total_duration_penalty,t_initial, t_freeze, alpha, iterations_per_temp, max_iterations)
