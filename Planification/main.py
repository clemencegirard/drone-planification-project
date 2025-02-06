###############  Import section ###################

from adjacency_matrix import *
from warehouse_builder import load_config,build_warehouse

###############  Code principal ###################

# Load confi
warehouses_config, category_mapping = load_config()

# Sélectionner un entrepôt
warehouse_name = "one_level_line_warehouse"

# Build warehouse
warehouse_3d = build_warehouse(warehouse_name, warehouses_config)

#False by default. If True, will display the warehouse in a plot
warehouse_3d.display(True)

#Generates the adjacency matrix
final_adjacency_matrix = main_adjacency(warehouse_3d, category_mapping)

#Save the adjacency matrix generated for the warehouse in the folder AMatrix as a csv file
save_adj_matrix(final_adjacency_matrix, warehouse_name)


print(final_adjacency_matrix)