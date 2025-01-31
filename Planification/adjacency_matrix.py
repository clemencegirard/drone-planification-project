import logging
from tqdm import tqdm
import numpy as np
from pathlib import Path

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

def generate_adjacency_matrix_by_blocks(warehouse_3d, coordinates, block_size=100):
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

    for i in tqdm(range(0, n, block_size), desc='Generating Manhattan distance for block i'):
        logging.info(f"Adjacency matrice for block number {i} is being generated...")


        for j in range(i, n, block_size):
            i_end = min(i + block_size, n)
            j_end = min(j + block_size, n)

            # Distance calculation for blocks
            for k in range(i, i_end):
                for l in range(j, j_end): #Using the symetry property of the adjacency matrix to compute only 1/2 the distances
                    if k != l:
                        adj_matrix[k, l] = warehouse_3d.compute_manhattan_distance(
                            coordinates[k], coordinates[l]
                        )
                        adj_matrix[l, k] = adj_matrix[k, l]

    return adj_matrix

def update_with_inter_category_distances(named_coordinates_dict, warehouse_3d, global_matrix, category_positions, category1, category2):
    """_summary_

    Args:
        category1 (_type_): _description_
        category2 (_type_): _description_
    """    
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
    category_positions = {}  # Initialise comme un dictionnaire

    # Placer chaque matrice dans la matrice globale
    for category, adj_matrix in adj_matrices.items():
        size = adj_matrix.shape[0]
        global_matrix[current_position:current_position + size,
                      current_position:current_position + size] = adj_matrix
        category_positions[category] = (current_position, current_position + size)
        current_position += size
    
    update_with_inter_category_distances(named_coordinates_dict, warehouse_3d, global_matrix, category_positions, 'object', 'checkpoint')
    update_with_inter_category_distances(named_coordinates_dict, warehouse_3d, global_matrix, category_positions, 'storage_line', 'checkpoint')

    return global_matrix

def main_adjacency(warehouse_3d, category_mapping):
    warehouse_mat = warehouse_3d.mat

    values_to_extract = list(category_mapping.keys())  # [0, 2, 3, 4]
    full_coordinates_dict = extract_coordinates(warehouse_mat, values_to_extract)

    named_coordinates_dict = {
        category_mapping[value]: coords
        for value, coords in full_coordinates_dict.items()
    }

    # Creation of one ajacency matrix per category
    logging.info("Generating adjacency matrices...")
    adj_matrices = {}

    for name, coordinates in named_coordinates_dict.items():

        if name == 'empty':
            # Adjacency matrices for empty is made of zeros, it cannot be reach by a drone.
            logging.info(f"Generating the adjacency matrice for category {name} ...")
            n = len(coordinates)
            adj_matrices[name] = np.zeros((n, n), dtype=float)
            logging.info(f"Adjacency matrice for category {name} generated.")

        else:
            # Display coordinates of every points for each category
            print(f"Coordonnées pour {name} : {coordinates}", len(coordinates))

            logging.info(f"Generating the adjacency matrice for category {name} ...")

            adj_matrices[name] = generate_adjacency_matrix_by_blocks(
                warehouse_3d, coordinates, block_size=25
            )
            logging.info(f"Adjacency matrice for category {name} generated.")

    adjacency_matrix = assemble_global_adjacency_matrix(named_coordinates_dict, warehouse_3d, adj_matrices)

    return adjacency_matrix


def save(adjacency_matrix : np.ndarray, warehouse_name : str):

    current_path = Path(__file__).parent.resolve()

    amatrix_dir = current_path / "AMatrix"

    #Creates the AMatrix file if it doesn't exists already
    amatrix_dir.mkdir(parents=True, exist_ok=True)

    file_name = f'AM_{warehouse_name}.csv'
    file_path = amatrix_dir / file_name

    # Save the  matrix in a csv file
    np.savetxt(file_path, adjacency_matrix, delimiter=",")
    print(f"Adjacency matrix saved to {file_path}")