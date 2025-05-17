# Warehouse generation

This module is dedicated to the creation and modelisation of our warehouses. It includes scripts that contain class for object oriented programming and also a builder.


## Warehouse 3D Management System

`warehouse.py` est le module principal de gestion d'un entrepôt en 3D. Il permet de modéliser un entrepôt en trois dimensions, de gérer les zones de stockage, les objets, les points de passage et les stations de recharge. Ce module est essentiel pour la planification et l'optimisation des déplacements des drones dans l'entrepôt.

### Core Features

#### Initialisation et Configuration

- **`Warehouse3D(name, rows, cols, height, mat_capacity)`**  
  Initialise un entrepôt 3D avec les dimensions spécifiées et une capacité de stockage maximale.

#### Visualisation et Sauvegarde

- **`display(display=False)`**  
  Affiche une représentation graphique de l'entrepôt avec les différents éléments (étagères, objets, points de passage, etc.).

- **`save_warehouse_plot(fig)`**  
  Sauvegarde l'affichage de l'entrepôt sous forme d'image dans un dossier dédié.

#### Gestion des Structures et du Stockage

- **`add_shelf(height, top_left, top_right, bottom_left, bottom_right)`**  
  Ajoute une étagère à un niveau donné dans l'entrepôt.

- **`add_storage_line(height, c1, c2)`**  
  Définit une ligne de stockage sur une étagère existante.

- **`add_object(rows, col, level)`**  
  Place un objet à un emplacement de stockage défini.

#### Gestion des Points de Départ et d'Arrivée

- **`add_start_mat(rows, col, level)`**  
  Définit une zone de départ pour les drones.

- **`add_finish_mat(rows, col, level)`**  
  Définit une zone d'arrivée pour les drones.

- **`add_charging_station(rows, col, level)`**  
  Ajoute une station de recharge pour les drones.

#### Gestion des Points de Passage et Connexions

- **`add_checkpoint(info)`**  
  Ajoute un point de passage pour faciliter la navigation des drones.

- **`connect_checkpoints(couple)`**  
  Connecte deux points de passage pour définir un chemin possible dans l'entrepôt.

#### Erreurs et Logs

- **`WarehouseError(Exception)`**  
  Exception personnalisée pour signaler une erreur de configuration ou d'exécution dans l'entrepôt.

- **`logging.basicConfig(...)`**  
  Configuration du système de logs pour suivre les événements et erreurs.


## Warehouse builder

### `warehouse_builder.py` - Configuration et Construction d'Entrepôts en 3D

Le module `warehouse_builder.py` est conçu pour charger une configuration JSON et construire dynamiquement des entrepôts en 3D à l'aide du module `Warehouse3D`. Il permet d'automatiser la création d'entrepôts en fonction des spécifications définies dans un fichier de configuration.

---

### Fonctionnalités principales

#### **Chargement de la configuration**

- **`load_config_warehouse(config_path="config_warehouses.json")`**  
  Charge le fichier JSON contenant les informations des entrepôts et la correspondance des catégories.

  - **Entrée** : Chemin vers le fichier de configuration (`config_warehouses.json`).
  - **Sortie** : 
    - `warehouses_config` : Dictionnaire contenant les configurations des entrepôts.
    - `category_mapping` : Dictionnaire mappant les catégories d'objets.

#### **Construction d'un entrepôt à partir de la configuration**

- **`build_warehouse(warehouse_name, warehouses_config)`**  
  Construit un entrepôt en 3D en fonction du nom spécifié.

  - **Entrée** :  
    - `warehouse_name` : Nom de l'entrepôt à construire.
    - `warehouses_config` : Configuration JSON des entrepôts.

  - **Sortie** : Objet `Warehouse3D` correspondant à l'entrepôt construit.

---

###  Processus de construction d'un entrepôt

1. **Initialisation**  
   - Vérifie si l'entrepôt existe dans la configuration.  
   - Extrait les dimensions et la capacité de stockage.  
   - Crée une instance de `Warehouse3D`.

2. **Ajout des structures**  
   - Ajout des étagères (`add_shelf`).  
   - Ajout des lignes de stockage (`add_storage_line` pour les horizontales et verticales).  

3. **Ajout des éléments spécifiques**  
   - Ajout des objets stockés (`add_object`).  
   - Ajout des points de passage (`add_checkpoint`).  
   - Connexion des points de passage (`connect_checkpoints`).  
   - Définition des zones de départ (`add_start_mat`) et d'arrivée (`add_finish_mat`).  
   - Ajout des stations de recharge (`add_charging_station`).  

4. **Retourne l'entrepôt construit**  

