###############  Import section ###################

from adjacency_matrix import *
from warehouse_builder import load_config,build_warehouse


###############  Code principal ###################

# Charger la configuration
warehouses_config, category_mapping = load_config()


# Sélectionner un entrepôt
warehouse_name = "warehouse2"

# Construire l'entrepôt
warehouse_3d = build_warehouse(warehouse_name, warehouses_config)

#Visualiser l'entrepot
warehouse_3d.display(True)

#Get adjacency matrix
final_adjacency_matrix = main_adjacency(warehouse_3d, category_mapping)

#save it
save(final_adjacency_matrix,warehouse_name)

print(final_adjacency_matrix)


