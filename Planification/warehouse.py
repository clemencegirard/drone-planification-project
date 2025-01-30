import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import logging

# Logs configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Exception personnalisée pour arrêter le programme en cas d'erreur dans le warehouse
class WarehouseError(Exception):
    pass

class Warehouse3D:
    def __init__(self, rows: int, cols: int, height: int):
        self.rows = rows
        self.cols = cols
        self.height = height
        self.mat = np.zeros((self.rows, self.cols, self.height))

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
                "Passage Point"
            ]
            colors = ["white", "gray", "orange", "red", "blue"]

            # Create legend handles
            handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10) for c in colors]
            fig.legend(handles, labels, loc="upper right", fontsize=10)

            plt.show()

    def print_mat(self):
        return print(self.mat)

    def add_shelf(self, height: int, top_left: tuple, top_right: tuple, bottom_left: tuple, bottom_right: tuple):
        if not (0 <= height < self.height):
            logging.warning(f"Level {height} out of bounds")
            raise WarehouseError(f"Level {height} out of bounds")  # Levée de l'exception
            return

        # Check if rectangle dimensions are valid
        if abs(top_left[1] - top_right[1]) == abs(bottom_left[1] - bottom_right[1]):
            width = abs(top_left[1] - top_right[1])
        else:
            logging.warning("Not a rectangle")
            raise WarehouseError("Not a rectangle")  # Levée de l'exception
            return

        if abs(top_left[0] - bottom_left[0]) == abs(top_right[0] - bottom_right[0]):
            height_rect = abs(top_left[0] - bottom_left[0])
        else:
            logging.warning("Not a rectangle")
            raise WarehouseError("Not a rectangle")  # Levée de l'exception
            return

        # Fill in the lower levels
        for h in range(height + 1):  # Includes the specified level and all below it
            for i in range(height_rect+1):
                for j in range(width+1):
                    self.mat[i + top_left[0], j + top_left[1], h] = 1


    def add_storage_line(self, height: int, c1: tuple, c2: tuple):
        if not (0 <= height < self.height):  # Check bounds
            logging.warning(f"Level {height} out of bounds")
            raise WarehouseError(f"Level {height} out of bounds")  # Levée de l'exception
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
                        raise WarehouseError("The specified line is out of storage")  # Levée de l'exception
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
                        raise WarehouseError("The specified line is out of storage")  # Levée de l'exception
                        return
            # fill it
            if is_storage:
                for h in range(height + 1):
                    for j in range(abs(x2 - x1) + 1):
                        self.mat[min(x1, x2) + j, y1, h] = 2

        else:
            logging.warning("Not a storage line")
            raise WarehouseError("Not a storage line")  # Levée de l'exception
            return

    def add_object(self, rows: int, col: int, level: int):

        if self.mat[rows, col, level] == 2:
            self.mat[rows, col, level] = 3
        else:
            logging.warning("There is no shelf here")
            raise WarehouseError("There is no shelf here")  # Levée de l'exception

    def add_checkpoint(self, rows: int, col: int, level: int):
        if self.mat[rows, col, level] == 0:
            self.mat[rows, col, level] = 4
        else:
            logging.warning("There is an obstacle here")
            raise WarehouseError("There is an obstacle here")  # Levée de l'exception

    def compute_manhattan_distance(self, c1: tuple, c2: tuple) -> int:
        x1, y1, z1 = c1
        x2, y2, z2 = c2

        if not (0 <= x1 < self.rows and 0 <= y1 < self.cols and 0 <= z1 < self.height):
            logging.warning("Start point is out of warehouse bounds.")
            raise WarehouseError("Start point is out of warehouse bounds.")  # Levée de l'exception
            return float('inf')

        if not (0 <= x2 < self.rows and 0 <= y2 < self.cols and 0 <= z2 < self.height):
            logging.warning("End point is out of warehouse bounds.")
            raise WarehouseError("End point is out of warehouse bounds.")  # Levée de l'exception
            return float('inf')

        # Check if points are valid
        if self.mat[x1, y1, z1] == 1 or self.mat[x2, y2, z2] == 1:
            logging.warning('Point does not correspond to a circulation zone.')
            raise WarehouseError('Point does not correspond to a circulation zone.')  # Levée de l'exception
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

