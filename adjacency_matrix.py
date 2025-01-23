from modelisation_3D import Warehouse3D
import numpy as np
import logging

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

    for i in range(0, n, block_size):
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

def assemble_global_adjacency_matrix(*matrices):
    """
    Assemble a global adjacency matrix from individual adjacency matrices provided as arguments.
    Parameters:
        matrices (list of np.ndarray): The adjacency matrices to be combined into a block-diagonal global matrix.

    Returns:
        np.ndarray: The global adjacency matrix combining all individual matrices as block matrices.
    """
    total_size = sum(matrix.shape[0] for matrix in matrices)
    global_matrix = np.zeros((total_size, total_size), dtype=float)

    current_position = 0

    for matrix in matrices:
        size = matrix.shape[0]  # Size of the current matrix
        global_matrix[current_position:current_position + size,
                      current_position:current_position + size] = matrix
        current_position += size

    return global_matrix


if __name__ == "__main__":

    #Build warehouse
    logging.info("Starting warehouse creation...")
    height = 3
    warehouse_3d = Warehouse3D(30, 30, height)

    # Add shelves at level 0
    
    # shelves
    logging.info("Adding shelves...")
    warehouse_3d.add_shelf(height, [2, 2], [2, 28], [5, 2], [5, 28])
    warehouse_3d.add_shelf(height, [2, 2], [2, 5], [20, 5], [20, 2])
    warehouse_3d.add_shelf(height, [2, 14], [2, 17], [20, 17], [20, 14])
    warehouse_3d.add_shelf(height, [2, 25], [2, 28], [20, 28], [20, 25])

    # horizontal storage line
    logging.info("Adding horizontal storage lines...")
    warehouse_3d.add_storage_line(height, [4, 5], [4, 13])
    warehouse_3d.add_storage_line(height, [4, 17], [4, 24])
    warehouse_3d.add_storage_line(height, [2, 3], [2, 26])

    # vertical storage line
    logging.info("Adding vertical storage lines...")
    warehouse_3d.add_storage_line(height, [3, 2], [19, 2])
    warehouse_3d.add_storage_line(height, [5, 4], [19, 4])
    warehouse_3d.add_storage_line(height, [5, 14], [19, 14])
    warehouse_3d.add_storage_line(height, [5, 16], [19, 16])
    warehouse_3d.add_storage_line(height, [5, 25], [19, 25])
    warehouse_3d.add_storage_line(height, [3, 27], [19, 27])


    # Add objects
    logging.info("Adding objects...")
    warehouse_3d.add_object(2, 6, 1)
    warehouse_3d.add_object(9, 4, 0)
    warehouse_3d.add_object(4, 20, 2)

    # Add checkpoints
    logging.info("Adding checkpoints...")
    warehouse_3d.add_checkpoint(20, 1, 0)
    warehouse_3d.add_checkpoint(20, 1, 1)
    warehouse_3d.add_checkpoint(20, 1, 2)

    warehouse_3d.add_checkpoint(20, 9, 0)
    warehouse_3d.add_checkpoint(20, 9, 1)
    warehouse_3d.add_checkpoint(20, 9, 2)

    warehouse_3d.add_checkpoint(10, 9, 0)
    warehouse_3d.add_checkpoint(10, 9, 1)
    warehouse_3d.add_checkpoint(10, 9, 2)

    warehouse_3d.add_checkpoint(20, 21, 0)
    warehouse_3d.add_checkpoint(20, 21, 1)
    warehouse_3d.add_checkpoint(20, 21, 2)

    warehouse_3d.add_checkpoint(10, 21, 0)
    warehouse_3d.add_checkpoint(10, 21, 1)
    warehouse_3d.add_checkpoint(10, 21, 2)

    warehouse_3d.add_checkpoint(20, 28, 0)
    warehouse_3d.add_checkpoint(20, 28, 1)
    warehouse_3d.add_checkpoint(20, 28, 2)

    warehouse_3d.add_checkpoint(1, 1, 0)
    warehouse_3d.add_checkpoint(1, 1, 1)
    warehouse_3d.add_checkpoint(1, 1, 2)

    warehouse_3d.add_checkpoint(1, 28, 0)
    warehouse_3d.add_checkpoint(1, 28, 1)
    warehouse_3d.add_checkpoint(1, 28, 2)

    category_mapping = {
        0: 'empty',
        1: 'shelf',
        2: 'storage_line',
        3: 'object',
        4: 'checkpoint',
    }

    warehouse_mat = warehouse_3d.mat
    values_to_extract = list(category_mapping.keys()).pop(1) # [0, 2, 3, 4]
    full_coordinates_dict = extract_coordinates(warehouse_mat, values_to_extract)

    named_coordinates_dict = {
        category_mapping[value]: coords
        for value, coords in full_coordinates_dict.items()
    }

    #Creation of one ajacency matrix per category
    logging.info("Generating adjacency matrices...")
    adj_matrices = {}
    for name, coordinates in named_coordinates_dict.items():

        if name == 'empty':
            #Adjacency matrices for empty is made of zeros, it cannot be reach by a drone.
            logging.info(f"Generating the adjacency matrice for category {name} ...")
            n = len(coordinates)
            adj_matrices[name] = np.zeros((n, n), dtype=float)
            logging.info(f"Adjacency matrice for category {name} generated.")

        else:
            #Display coordinates of every points for each category
            print(f"Coordonnées pour {name} : {coordinates}", len(coordinates))

            logging.info(f"Generating the adjacency matrice for category {name} ...")

            #adj_matrix_name = generate_adjacency_matrix(warehouse_3d, coordinates)

            adj_matrices[name] = generate_adjacency_matrix_by_blocks(
            warehouse_3d, coordinates, block_size=25
            )
            logging.info(f"Adjacency matrice for category {name} generated.")

    adjacency_matrix = assemble_global_adjacency_matrix(
        adj_matrices['empty'],
        adj_matrices['shelf'],
        adj_matrices['storage_line'],
        adj_matrices['object'],
        adj_matrices['checkpoint']
    )