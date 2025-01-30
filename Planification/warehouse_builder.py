import json
from warehouse import Warehouse3D
import logging


def load_config(config_path="Planification\\config_warehouses.json"):
    with open(config_path, "r") as f:
        config = json.load(f)

    # Séparer la config des entrepôts et le mapping des catégories
    warehouses_config = config["warehouses"]
    category_mapping = {int(k): v for k, v in config["category_mapping"].items()}  # 🔥 Conversion ici

    return warehouses_config, category_mapping


def build_warehouse(warehouse_name, warehouses_config):
    """Construit un entrepôt à partir du nom et de la configuration."""
    warehouse_data = warehouses_config.get(warehouse_name)

    if warehouse_data is None:
        raise KeyError(f"L'entrepôt '{warehouse_name}' n'existe pas dans la configuration.")


    logging.info(f"Starting {warehouse_name} creation...")

    # Création de l'entrepôt
    dimensions = warehouse_data["dimensions"]
    warehouse_3d = Warehouse3D(*dimensions)

    # Ajout des étagères
    logging.info("Adding shelves...")
    for shelf in warehouse_data["shelves"]:
        # Assurez-vous de passer la hauteur correctement, sans inclure dimensions[2]
        warehouse_3d.add_shelf(shelf[0], *shelf[1:])

    # Ajout des lignes de stockage
    logging.info("Adding storage lines...")
    for line in warehouse_data["storage_lines"]["horizontal"]:
        warehouse_3d.add_storage_line(line[0], *line[1:])
    for line in warehouse_data["storage_lines"]["vertical"]:
        warehouse_3d.add_storage_line(line[0], *line[1:])

    # Ajout des objets
    logging.info("Adding objects...")
    for obj in warehouse_data["objects"]:
        warehouse_3d.add_object(*obj)

    # Ajout des checkpoints
    logging.info("Adding checkpoints...")
    for checkpoint in warehouse_data["checkpoints"]:
        for level in range(dimensions[2]):  # Ajouter un checkpoint à chaque niveau
            warehouse_3d.add_checkpoint(*checkpoint, level)

    return warehouse_3d

