import logging
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

#Logs configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_coordinates(matrix, values) -> dict:
    """
    Parameters:
        matrix (np.ndarray): Warehouse matrix.
        values (list): List of values to look for (here [0, 1, 2, 3, 4]).
    Returns: Dictionnary where each key has a value and each value is a list of tuples (x, y, z).
    """
    coordinates = {value: [] for value in values}
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            for z in range(matrix.shape[2]):
                cell_value = matrix[x, y, z]
                if cell_value in values:
                    coordinates[cell_value].append((x, y, z))
    return coordinates

def generate_adjacency_matrix(warehouse_3d, coordinates):
    """
    Parameters
        warehouse_3d : modelisation_3D from the class Warehouse3D.
        coordinate (int): Coordinates of all the points from one category (being within {0,1,2,3,4}).
    Returns: Adjacency matrix of the set of points given using the Manhattan distance.
    """  
    n = len(coordinates)

    adj_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i, n):
            if i!=j:

                adj_matrix[i,j] = warehouse_3d.compute_manhattan_distance(
                    coordinates[i], coordinates[j]
                    )
                adj_matrix[j,i] = warehouse_3d.compute_manhattan_distance(
                    coordinates[i], coordinates[j]
                    )

    return adj_matrix

def generate_diagonal_checkpoints_adjmatrix(warehouse_3d, coordinates, block_size=25):
    """
    Generates an adajcency matrix by blocks to decrease running time.
    Parameters:
        warehouse_3d : Warehouse3D instance to call Manhattandistance function.
        coordinates : For one category, list of every points coordinates.
        block_size : Block size for the matrices.
    Returns:
        adj_matrix : Full adjacency matrix.
    """
    n = len(coordinates)
    adj_matrix = np.zeros((n, n), dtype=float)
    checkpoint_connection = warehouse_3d.checkpoints_graph

    for block_start in tqdm(range(0, n, block_size), desc="Processing block pairs"):
        block_end = min(block_start + block_size, n)

        for i in range(block_start, block_end):
            for j in range(block_start, block_end):
                if j in checkpoint_connection.get(i, set()):
                        print("coordonnées de (i,j): ", (i,j))
                        if i != j:
                            coord1 = coordinates[i]
                            coord2 = coordinates[j]
                            dist = warehouse_3d.compute_manhattan_distance(coord1, coord2)
                            adj_matrix[i, j] = dist
                            adj_matrix[j, i] = dist
                else:
                    print((i,j), "are not connected")

    return adj_matrix

def update_with_inter_category_distances(named_coordinates_dict, warehouse_3d, global_matrix, category_positions, category1, category2):
     
    start1, _ = category_positions[category1]
    start2, _ = category_positions[category2]
    
    coords1 = named_coordinates_dict[category1]
    coords2 = named_coordinates_dict[category2]

    for i, coord1 in enumerate(coords1):
        for j, coord2 in enumerate(coords2):
            dist = warehouse_3d.compute_manhattan_distance(coord1, coord2)
            global_matrix[start1 + i, start2 + j] = dist
            global_matrix[start2 + j, start1 + i] = dist


def assemble_global_adjacency_matrix(named_coordinates_dict, warehouse_3d, adj_matrices):
    """
    Assemble a global adjacency matrix from individual adjacency matrices provided as arguments.
    Parameters:
        named_coordinates_dict (dict): Coordinates for each category.
        warehouse_3d: Warehouse object.
        adj_matrices (list of np.ndarray): Individual adjacency matrices.

    Returns:
        np.ndarray: Global adjacency matrix combining all individual matrices as block matrices.
    """

    total_size = sum(matrix.shape[0] for matrix in adj_matrices.values())
    global_matrix = np.zeros((total_size, total_size), dtype=float)

    current_position = 0
    category_positions = {}

    # Place every diagonal matrice in the global matrix
    for category, adj_matrix in adj_matrices.items():
        size = adj_matrix.shape[0]
        global_matrix[current_position:current_position + size,
                      current_position:current_position + size] = adj_matrix
        category_positions[category] = (current_position, current_position + size)
        current_position += size
    
    #Add non-diagonal block matrices in the global matrix
    update_with_inter_category_distances(named_coordinates_dict, warehouse_3d, global_matrix, category_positions, 'object', 'checkpoint')
    update_with_inter_category_distances(named_coordinates_dict, warehouse_3d, global_matrix, category_positions, 'storage_line', 'checkpoint')
    update_with_inter_category_distances(named_coordinates_dict, warehouse_3d, global_matrix, category_positions, 'start_mat', 'finish_mat')
    update_with_inter_category_distances(named_coordinates_dict, warehouse_3d, global_matrix, category_positions, 'start_mat', 'checkpoint')
    update_with_inter_category_distances(named_coordinates_dict, warehouse_3d, global_matrix, category_positions, 'finish_mat', 'checkpoint')

    return global_matrix

def main_adjacency(warehouse_3d, category_mapping):
    warehouse_mat = warehouse_3d.mat

    values_to_extract = list(category_mapping.keys())
    full_coordinates_dict = extract_coordinates(warehouse_mat, values_to_extract)

    named_coordinates_dict = {
        category_mapping[value]: coords
        for value, coords in full_coordinates_dict.items()
    }

    #Generates a mapper between coordinates and its position in the adjacency matrix
    coordinate_to_index = {}

    index = 1
    for coordinates_list in full_coordinates_dict.values():
        for coord in coordinates_list:
            coordinate_to_index[coord] = index
            index += 1

    # Creation of one ajacency matrix per category
    logging.info("Generating adjacency matrices...")
    adj_matrices = {}

    for name, coordinates in named_coordinates_dict.items():

        if name == "checkpoint":
            #Generates the diagonal matrice
            logging.info(f"Generating the adjacency matrice for category {name} ...")

            adj_matrices[name] = generate_diagonal_checkpoints_adjmatrix(
                warehouse_3d, coordinates, block_size=25
            )
            logging.info(f"Diagonal adjacency matrice for category {name} generated.")

        else:
            # Diagonal adjacency matrices for every category except checkpoint is made of zeros.
            logging.info(f"Generating the adjacency matrice for category {name} ...")
            n = len(coordinates)
            adj_matrices[name] = np.zeros((n, n), dtype=float)
            logging.info(f"Adjacency matrice for category {name} generated.")

    #Assemble diagonal and non diagonal block matrices
    adjacency_matrix = assemble_global_adjacency_matrix(named_coordinates_dict, warehouse_3d, adj_matrices)

    return adjacency_matrix, coordinate_to_index


def save_adj_matrix(adjacency_matrix : np.ndarray, warehouse_name : str):

    current_path = Path(__file__).parent.resolve()

    amatrix_dir = current_path / "AMatrix"

    #Creates the AMatrix file if it doesn't exists already
    amatrix_dir.mkdir(parents=True, exist_ok=True)

    file_name = f'AM_{warehouse_name}.csv'
    file_path = amatrix_dir / file_name

    # Save the  matrix in a csv file
    np.savetxt(file_path, adjacency_matrix, delimiter=",")
    print(f"Adjacency matrix saved to {file_path}")


def save_warehouse_plot(fig, warehouse_name: str):
    current_path = Path(__file__).parent.resolve()
    warehouse_plot_dir = current_path / "warehouse_plot"
    warehouse_plot_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{warehouse_name}.png"
    file_path = warehouse_plot_dir / file_name

    fig.savefig(file_path)
    print(f"Warehouse plot saved to {file_path}")
    plt.show()