import json
from pathlib import Path
from Warehouse.warehouse import Warehouse3D
import logging

def load_config_warehouse(config_path="config_warehouses.json"):
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
    mat_capacity = warehouse_data["mat_capacity"][0]
    warehouse_3d = Warehouse3D(warehouse_name, *dimensions, mat_capacity)

    # Adding shelves
    logging.info("Adding shelves...")
    for shelf in warehouse_data["shelves"]:
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
        warehouse_3d.add_checkpoint(checkpoint)

    # Connect checkpoints
    logging.info("Connecting checkpoints...")
    for connection in warehouse_data["checkpoint_connection"]:
        warehouse_3d.connect_checkpoints(connection)

    # Adding start mat
    logging.info("Adding start mat...")
    for start_mat in warehouse_data["start_mat"]:
        warehouse_3d.add_start_mat(*start_mat)

    # Adding finish mat
    logging.info("Adding finish mat...")
    for finish_mat in warehouse_data["finish_mat"]:
        warehouse_3d.add_finish_mat(*finish_mat)

    # Adding charging station
    logging.info("Adding charging station ...")
    for charging_station in warehouse_data["charging_station"]:
        warehouse_3d.add_charging_station(*charging_station)

    return warehouse_3d

