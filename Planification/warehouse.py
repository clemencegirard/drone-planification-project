import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import logging
from pathlib import Path
import networkx as nx

# Logs configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Personnalised exception to stop the code when error in the warehouse
class WarehouseError(Exception):
    pass

class Warehouse3D:
    def __init__(self, name: str, rows: int, cols: int, height: int, mat_capacity :int):

        self.name = name
        self.rows = rows
        self.cols = cols
        self.height = height
        self.mat_capacity = mat_capacity
        self.mat = np.zeros((self.rows, self.cols, self.height))
        self.checkpoints_graph = {}

    def save_warehouse_plot(self, fig):
        current_path = Path(__file__).parent.resolve()
        warehouse_plot_dir = current_path / "warehouse_plot"
        warehouse_plot_dir.mkdir(parents=True, exist_ok=True)

        file_name = f"{self.name}.png"
        file_path = warehouse_plot_dir / file_name

        fig.savefig(file_path)
        print(f"Warehouse plot saved to {file_path}")

    def display(self, display=False):
        if display:
            fig, axes = plt.subplots(1, self.height, figsize=(5 * self.height, 5))
            for h in range(self.height):
                ax = axes[h] if self.height > 1 else axes
                ax.imshow(self.mat[:, :, h], cmap="Greys", origin="upper")

                # Display orange circles for cells where self.mat == 2 (object location)
                # and red circles for cells where self.mat == 3 (object)
                for x in range(self.rows):
                    for y in range(self.cols):
                        if self.mat[x, y, h] == 2:
                            ax.add_patch(plt.Circle((y, x), 0.4, color="orange", ec="black"))
                        elif self.mat[x, y, h] == 3:
                            ax.add_patch(plt.Circle((y, x), 0.4, color="red", ec="black"))
                        elif self.mat[x, y, h] == 4:
                            ax.add_patch(plt.Circle((y, x), 0.4, color="blue", ec="black"))
                        elif self.mat[x, y, h] == 5:
                            ax.add_patch(plt.Circle((y, x), 0.4, color="yellow", ec="black"))
                        elif self.mat[x, y, h] == 6:
                            ax.add_patch(plt.Circle((y, x), 0.4, color="green", ec="black"))

                ax.set_title(f"Level {h}")
                ax.grid(which="both", color="black", linewidth=0.4)
                ax.set_xticks(range(self.cols))
                ax.set_yticks(range(self.rows))

            # Add legend
            labels = [
                "Free Zone",
                "Shelf",
                "Object Location",
                "Object Presence",
                "Passage Point",
                "Start Mat",
                "Finish Mat"
            ]
            colors = ["white", "gray", "orange", "red", "blue","yellow","green"]

            # Create legend handles
            handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10) for c in colors]
            fig.legend(handles, labels, loc="upper right", fontsize=10)
            self.save_warehouse_plot(fig)
            plt.show()

    def print_mat(self):
        return print(self.mat)

    def add_shelf(self, height: int, top_left: tuple, top_right: tuple, bottom_left: tuple, bottom_right: tuple):
        if not (0 <= height < self.height):
            logging.warning(f"Level {height} out of bounds")
            raise WarehouseError(f"Level {height} out of bounds")  # Raise error
            return

        # Check if rectangle dimensions are valid
        if abs(top_left[1] - top_right[1]) == abs(bottom_left[1] - bottom_right[1]):
            width = abs(top_left[1] - top_right[1])
        else:
            logging.warning("Not a rectangle")
            raise WarehouseError("Not a rectangle")  # Raise error
            return

        if abs(top_left[0] - bottom_left[0]) == abs(top_right[0] - bottom_right[0]):
            height_rect = abs(top_left[0] - bottom_left[0])
        else:
            logging.warning("Not a rectangle")
            raise WarehouseError("Not a rectangle")  # Raise error
            return

        # Fill in the lower levels
        for h in range(height + 1):  # Includes the specified level and all below it
            for i in range(height_rect+1):
                for j in range(width+1):
                    self.mat[i + top_left[0], j + top_left[1], h] = 1

    def add_storage_line(self, height: int, c1: tuple, c2: tuple):
        if not (0 <= height < self.height):  # Check bounds
            logging.warning(f"Level {height} out of bounds")
            raise WarehouseError(f"Level {height} out of bounds")  # Raise error
            return

        x1, y1 = c1
        x2, y2 = c2
        is_storage = False

        if x1 == x2:  # Horizontal line
            #check if shelf is settled
            for h in range(height + 1):  # Fill all levels up to the specified height
                for j in range(abs(y2 - y1) + 1):
                    if self.mat[x1, min(y1, y2) + j, h] == 1 or self.mat[x1, min(y1, y2) + j, h] == 2:
                        is_storage = True
                    else:
                        logging.warning("The specified line is out of storage")
                        raise WarehouseError("The specified line is out of storage")  # Raise error
                        return
            #fill it
            if is_storage:
                for h in range(height + 1):
                    for j in range(abs(y2 - y1) + 1):
                        self.mat[x1, min(y1, y2) + j, h] = 2

        elif y1 == y2:  # Vertical line
            # check if shelf is settled
            for h in range(height+1):  # Fill all levels up to the specified height
                for j in range(abs(x2 - x1) + 1):
                    if self.mat[min(x1, x2) + j, y1, h] == 1 or self.mat[min(x1, x2) + j, y1, h] == 2:
                        is_storage = True
                    else:
                        logging.warning("The specified line is out of storage")
                        raise WarehouseError("The specified line is out of storage")  # Raise error
                        return
            # fill it
            if is_storage:
                for h in range(height + 1):
                    for j in range(abs(x2 - x1) + 1):
                        self.mat[min(x1, x2) + j, y1, h] = 2

        else:
            logging.warning("Not a storage line")
            raise WarehouseError("Not a storage line")  # Raise error
            return

    def add_object(self, rows: int, col: int, level: int):

        if self.mat[rows, col, level] == 2:
            self.mat[rows, col, level] = 3
        else:
            logging.warning("There is no shelf here")
            raise WarehouseError("There is no shelf here")  # Raise error

    def add_start_mat(self, rows: int, col: int, level: int):

        if self.mat[rows, col, level] == 0:
            self.mat[rows, col, level] = 5
        else:
            logging.warning("There is no free space here for start mat")
            raise WarehouseError("There is no free space here for start mate")  # Raise error

    def add_finish_mat(self, rows: int, col: int, level: int):

        if self.mat[rows, col, level] == 0:
            self.mat[rows, col, level] = 6
        else:
            logging.warning("There is no free space here for finish mat")
            raise WarehouseError("There is no free space here for finish mate")  # Raise error

    def add_checkpoint(self, info : tuple):
        row,col,level = info[0]
        checkpoint_id = info[1]
        if self.mat[row, col, level] == 0:
            self.mat[row, col, level] = 4
            self.checkpoints_graph[checkpoint_id] = set()  # Initialise les connexions vides
        else:
            logging.warning("Il y a un obstacle ici")
            raise WarehouseError("Il y a un obstacle ici")

    def connect_checkpoints(self, couple : tuple):
        cp1,cp2 = couple
        if cp1 in self.checkpoints_graph and cp2 in self.checkpoints_graph:
            self.checkpoints_graph[cp1].add(cp2)  # Connexion dirigée
        else:
            logging.warning("Checkpoint invalide")
            raise WarehouseError("Checkpoint invalide")

    def can_move_between_checkpoints(self, cp1: int, cp2: int) -> bool:
        return cp2 in self.checkpoints_graph.get(cp1, set())

    def show_graph(self, show=False):

        if show:
            # Créer un graphe orienté
            G = nx.DiGraph()

            # Ajouter les nœuds
            num_nodes = len(self.checkpoints_graph)

            for key, values in self.checkpoints_graph.items():
                G.add_node(key)
                for value in values:
                    G.add_edge(key, value)

            # Position des nœuds pour visualiser le graphe (arrangés en cercle)
            pos = nx.spring_layout(G)  # 'spring_layout' donne une disposition agréable

            # Dessiner le graphe
            plt.figure(figsize=(10, 8))
            nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue", font_size=10, font_weight="bold",
                    arrows=True)

            # Ajouter les étiquettes des poids
            labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

            plt.title("Graphe des temps entre points")

            return plt.show()


    def compute_manhattan_distance(self, c1: tuple, c2: tuple) -> int:
        x1, y1, z1 = c1
        x2, y2, z2 = c2

        if not (0 <= x1 < self.rows and 0 <= y1 < self.cols and 0 <= z1 < self.height):
            logging.warning("Start point is out of warehouse bounds.")
            raise WarehouseError("Start point is out of warehouse bounds.")  # Raise error
            return float('inf')

        if not (0 <= x2 < self.rows and 0 <= y2 < self.cols and 0 <= z2 < self.height):
            logging.warning("End point is out of warehouse bounds.")
            raise WarehouseError("End point is out of warehouse bounds.")  # Raise error
            return float('inf')

        # Check if points are valid
        if self.mat[x1, y1, z1] == 1 or self.mat[x2, y2, z2] == 1:
            logging.warning('Point does not correspond to a circulation zone.')
            raise WarehouseError('Point does not correspond to a circulation zone.')  # Raise error
            return float('inf')  # No path possible

        # Possible directions for movement (up, down, left, right, up-z, down-z)
        directions = [
            (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1),
        ]

        # Initialize BFS queue and visited cells list
        queue = deque([(x1, y1, z1, 0)])  # (x, y, z, distance)
        visited = np.zeros_like(self.mat, dtype=bool)

        # Mark the start point as visited
        visited[x1, y1, z1] = True

        while queue:
            x, y, z, dist = queue.popleft()

            # If we reach the end point, return the distance
            if (x, y, z) == (x2, y2, z2):
                return dist

            # Explore neighbors
            for dx, dy, dz in directions:
                nx, ny, nz = x + dx, y + dy, z + dz

                # Check if neighbor is within bounds, not visited, and free
                if (0 <= nx < self.rows and 0 <= ny < self.cols and 0 <= nz < self.height and
                        not visited[nx, ny, nz] and self.mat[nx, ny, nz] != 1):
                    visited[nx, ny, nz] = True  # Mark as visited
                    queue.append((nx, ny, nz, dist + 1))  # Add neighbor to queue

        # If we exit the loop without finding the end point, no path is possible
        return float('inf')

class Object:
    def __init__(self, id: str, is_on_shelf: bool, row: int, col: int, height: int):
        self.id = id
        self.is_on_shelf = is_on_shelf
        self.row = row
        self.col = col
        self.height = height

    def move_to(self, row: int, col: int, height: int):
        self.row = row
        self.col = col
        self.height = height
