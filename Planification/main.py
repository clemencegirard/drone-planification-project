###############  Import section ###################

from adjacency_matrix import *
from warehouse_builder import load_config,build_warehouse
from bellman import main_bellman

###############  Code principal ###################

# Load confi
warehouses_config, category_mapping = load_config()

# Sélectionner un entrepôt
warehouse_name = "three_level_line_warehouse"

# Build warehouse
warehouse_3d = build_warehouse(warehouse_name, warehouses_config)

#False by default. If True, will display the warehouse in a plot
warehouse_3d.display()
warehouse_3d.show_graph()

#Generates the adjacency matrix
final_adjacency_matrix, coordinate_to_index = main_adjacency(warehouse_3d, category_mapping)

print(final_adjacency_matrix)
print("Dictionnary to map a point coordinates and its position in the adjacency matrix :", coordinate_to_index)

#Save the adjacency matrix generated for the warehouse in the folder AMatrix as a csv file
save_adj_matrix(final_adjacency_matrix, warehouse_name)

print("\n(")
#Call Bellman algorithm
final_adjacency_matrix_2 = main_bellman(final_adjacency_matrix)

print(final_adjacency_matrix_2)