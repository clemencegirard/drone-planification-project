import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from Evitement.avoidance import *

def metropolis_acceptance(current_cost, new_cost, temp):
    if new_cost < current_cost:
        return True
    return random.uniform(0, 1) < np.exp((current_cost - new_cost) / temp)

def simulated_annealing(planning: Dict[str, pd.DataFrame], t_initial: float = 1000, t_freeze: float = 0.1, alpha: float = 0.9, iterations_per_temp = 500):

    temp = t_initial

    # Intitialisation
    current_planning = planning.copy()
    current_cost = compute_cost(current_planning)

    # Keep the best solution
    best_planning = current_planning

    pbar = tqdm(total=int(np.log(t_freeze / t_initial) / np.log(alpha)), desc="Simulated Annealing")

    while temp > t_freeze :
        # Explore solutions at constant temperature
        for _ in range(iterations_per_temp) :
            direct_collisions, calculated_collisions = count_direct_collisions(current_planning), len(count_calculated_collisions(current_planning))

            if not (direct_collisions.empty and calculated_collisions.empty) :
                new_planning = fix_collisions(planning, direct_collisions, calculated_collisions)
            else :
                new_planning = make_new_planning(planning)

            # Compare costs
            new_cost = compute_cost(new_planning)

            if metropolis_acceptance(current_cost, new_cost, temp) :
                current_planning, current_cost = new_planning, new_cost
            
            if new_cost < current_cost :
                best_planning = new_planning 
        # Cool the temperature
        temp = temp * alpha
        pbar.update(1)
    pbar.close()

    return best_planning

# Recursively finds an optimal solution that respects the constraints.
def find_optimal_solution(planning: Dict[str, pd.DataFrame], t_initial: float, t_freeze: float, alpha: float, iterations_per_temp, max_iterations: int):
    direct_collisons, calculated_collisions = count_direct_collisions(planning), count_calculated_collisions(planning)
    if direct_collisons.empty and calculated_collisions.empty :
        return planning, compute_cost(planning), True
    elif max_iterations == 0 :
        return planning, compute_cost(planning), False
    new_planning = simulated_annealing(planning, t_initial, t_freeze, alpha, iterations_per_temp)
    max_iterations -= 1
    return find_optimal_solution(new_planning, t_initial, t_freeze, alpha, iterations_per_temp, max_iterations)