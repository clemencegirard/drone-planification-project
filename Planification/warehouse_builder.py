import json
from pathlib import Path
from warehouse import Warehouse3D
import logging


def load_config(config_path="config_warehouses.json"):
    config_file = Path(__file__).parent / config_path
    with open(config_file, "r") as f:
        config = json.load(f)

    # Separate warehouse and category mapping config
    warehouses_config = config["warehouses"]
    category_mapping = {int(k): v for k, v in config["category_mapping"].items()}  # 🔥 Converting here

    return warehouses_config, category_mapping


def build_warehouse(warehouse_name, warehouses_config):
    """Construit un entrepôt à partir du nom et de la configuration."""
    warehouse_data = warehouses_config.get(warehouse_name)

    if warehouse_data is None:
        raise KeyError(f"L'entrepôt '{warehouse_name}' n'existe pas dans la configuration.")


    logging.info(f"Starting {warehouse_name} creation...")

    # Warehouse creation
    dimensions = warehouse_data["dimensions"]
    warehouse_3d = Warehouse3D(*dimensions)

    # Adding shelves
    logging.info("Adding shelves...")
    for shelf in warehouse_data["shelves"]:
        # Switch correctly to next height, without including dimensions [2]
        warehouse_3d.add_shelf(shelf[0], *shelf[1:])

    # Adding storage lines
    logging.info("Adding storage lines...")
    for line in warehouse_data["storage_lines"]["horizontal"]:
        warehouse_3d.add_storage_line(line[0], *line[1:])
    for line in warehouse_data["storage_lines"]["vertical"]:
        warehouse_3d.add_storage_line(line[0], *line[1:])

    # Adding objects
    logging.info("Adding objects...")
    for obj in warehouse_data["objects"]:
        warehouse_3d.add_object(*obj)

    # Adding checkpoints
    logging.info("Adding checkpoints...")
    for checkpoint in warehouse_data["checkpoints"]:
        for level in range(dimensions[2]):  # Adding one checkpoint at every level
            warehouse_3d.add_checkpoint(*checkpoint, level)

    # Adding start mat
    logging.info("Adding start mat...")
    for start_mat in warehouse_data["start_mat"]:
        warehouse_3d.add_start_mat(*start_mat)

    # Adding start mat
    logging.info("Adding finish mat...")
    for finish_mat in warehouse_data["finish_mat"]:
        warehouse_3d.add_finish_mat(*finish_mat)


    return warehouse_3d

