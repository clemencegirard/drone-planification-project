## warehouse.py

`warehouse.py` is one of the most important scripts in the project since it is used to design a tailor-made warehouse. This file contains several classes:

### Classes

1. **WarehouseError**
   - A custom exception class used to handle errors specific to the warehouse operations, such as invalid coordinates or obstacles.

2. **Warehouse3D**
   - The main class that represents a 3D warehouse. It includes methods to manage the warehouse layout, add shelves, storage lines, objects, and more. Key functionalities include:
     - **Initialization**: Creates a 3D matrix to represent the warehouse layout.
     - **Adding Shelves**: Allows the addition of shelves at specified levels and coordinates.
     - **Adding Storage Lines**: Marks storage lines on shelves for object placement.
     - **Adding Objects**: Places objects on shelves or designated storage locations.
     - **Adding Checkpoints**: Defines checkpoints for navigation and connects them in a graph.
     - **Pathfinding**: Uses BFS (Breadth-First Search) to compute the shortest path between two points in the warehouse.
     - **Visualization**: Provides methods to display the warehouse layout and checkpoint graph.

3. **Object**
   - Represents an object in the warehouse. It includes attributes like `id`, `is_on_shelf`, and coordinates (`row`, `col`, `height`). It also provides a method to move the object to a new location.

### Key Features

- **3D Warehouse Representation**: The warehouse is represented as a 3D matrix, allowing for multi-level storage and navigation.
- **Customizable Layout**: Users can add shelves, storage lines, objects, and checkpoints to design a warehouse tailored to their needs.
- **Pathfinding**: The BFS algorithm is used to compute the shortest path between two points, ensuring efficient navigation.
- **Visualization**: The warehouse layout and checkpoint graph can be visualized using `matplotlib` and `networkx`.

### Usage

To use the `warehouse.py` script, follow these steps:

1. **Initialize the Warehouse**:
   ```python
   warehouse = Warehouse3D(name="MyWarehouse", rows=10, cols=10, height=3, mat_capacity=100)


## adjacency_matrix.py

Tha adjacency matrix is generated from the warehouse matrix. The value at position (i, j) of the warehouse matrix correspond to the category of the case (i, j) in the warehouse, as defined in the dictionnary category_mapping.

The script `adjacency_matrix.py` goes through the warehouse matrix first on the z axis, then x axis, then y axis.

Adjacency matrix contains the Manhattan distances for these categories in order : 
- empty storage line (category 2)
- full storage line (category 3)
- checkpoint (catégorie 4)
- start mat (category 5)
- finish mat (category 6)
- charging station (category 7)

The script only calculates distances between checkpoint that are connected and relevant categories.

This way, it calculates the distance : 
- checkpoint and all its connected checkpoints
- checkpoints and storage line
- checkpoints and objects
- start mat and finish mat
- start mat and checkpoint
- finish mat and checkpoint
- (charging stations ?)

The script generates the adjacency matrix by block for these distances above only and assemble everything in a global matrix.


<img src="../Data_test/adjmatrix_schema.png" alt="Adjacency matrix" width="500" height="350">
<img src="../Data_test/U_warehouse.png" alt="Matrix" width="300" height="300">

