import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class Warehouse3D:
    def __init__(self, rows: int, cols: int, height: int):
        self.rows = rows
        self.cols = cols
        self.height = height
        self.mat = np.zeros((self.rows, self.cols, self.height))

    def display(self):
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

            ax.set_title(f"Level {h + 1}")
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
        if not (0 <= height <= self.height):
            return print(f"Level {height} out of bounds")

        # Check if rectangle dimensions are valid
        if abs(top_left[1] - top_right[1]) == abs(bottom_left[1] - bottom_right[1]):
            width = abs(top_left[1] - top_right[1])
        else:
            return print("Not a rectangle")
        if abs(top_left[0] - bottom_left[0]) == abs(top_right[0] - bottom_right[0]):
            height_rect = abs(top_left[0] - bottom_left[0])
        else:
            return print("Not a rectangle")

        # Fill in the lower levels
        for h in range(height):  # Includes the specified level and all below it
            for i in range(height_rect):
                for j in range(width):
                    self.mat[i + top_left[0], j + top_left[1], h] = 1

    def add_storage_line(self, height: int, c1: tuple, c2: tuple):
        if not (0 <= height <= self.height):  # Check bounds
            return print(f"Level {height} out of bounds")

        x1, y1 = c1
        x2, y2 = c2
        is_storage = False

        if x1 == x2:  # Vertical line
            for h in range(height):  # Fill all levels up to the specified height
                for j in range(abs(y2 - y1) + 1):
                    if self.mat[x1, min(y1, y2) + j, h] == 1 or self.mat[x1, min(y1, y2) + j, h] == 2:
                        is_storage = True
                    else:
                        return print("The specified line is out of storage")
                if is_storage:
                    for j in range(abs(y2 - y1) + 1):
                        self.mat[x1, min(y1, y2) + j, h] = 2

        elif y1 == y2:  # Horizontal line
            for h in range(height):  # Fill all levels up to the specified height
                for j in range(abs(x2 - x1) + 1):
                    if self.mat[min(x1, x2) + j, y1, h] == 1 or self.mat[min(x1, x2) + j, y1, h] == 2:
                        is_storage = True
                    else:
                        return print("The specified line is out of storage")
                if is_storage:
                    for j in range(abs(x2 - x1) + 1):
                        self.mat[min(x1, x2) + j, y1, h] = 2

        else:
            return print("Not a storage line")

    def add_object(self, rows: int, col: int, level: int):

        if self.mat[rows, col, level] == 2:
            self.mat[rows, col, level] = 3
        else:
            print("There is no shelf here")

    def add_checkpoint(self, rows: int, col: int, level: int):

        if self.mat[rows, col, level] == 0:
            self.mat[rows, col, level] = 4
        else:
            print("There is an obstacle here")


    def compute_manhattan_distance(self, c1, c2):

        x1, y1, z1 = c1
        x2, y2, z2 = c2

        if not (0 <= x1 < self.rows and 0 <= y1 < self.cols and 0 <= z1 < self.height):
            print("Start point is out of warehouse bounds.")
            return float('inf')

        if not (0 <= x2 < self.rows and 0 <= y2 < self.cols and 0 <= z2 < self.height):
            print("End point is out of warehouse bounds.")
            return float('inf')

        # Check if points are valid
        if self.mat[x1, y1, z1] not in [0, 4] or self.mat[x2, y2, z2] not in [0, 4]:
            print('Point does not correspond to a circulation zone.')
            return float('inf')  # No path possible

        # Possible directions for movement (up, down, left, right, up-z, down-z)
        directions = [
            (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)
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
                        not visited[nx, ny, nz] and self.mat[nx, ny, nz] in [0, 4]):
                    visited[nx, ny, nz] = True  # Mark as visited
                    queue.append((nx, ny, nz, dist + 1))  # Add neighbor to queue

        # If we exit the loop without finding the end point, no path is possible
        return float('inf')


if __name__ == "__main__":

    height = 3
    # Example usage
    warehouse_3d = Warehouse3D(30, 30, height)

    # Add shelves at level 0

    # shelves
    warehouse_3d.add_shelf(height, [2, 2], [2, 28], [5, 2], [5, 28])
    warehouse_3d.add_shelf(height, [2, 2], [2, 5], [20, 5], [20, 2])
    warehouse_3d.add_shelf(height, [2, 14], [2, 17], [20, 17], [20, 14])
    warehouse_3d.add_shelf(height, [2, 25], [2, 28], [20, 28], [20, 25])

    # horizontal storage line
    warehouse_3d.add_storage_line(height, [4, 5], [4, 13])
    warehouse_3d.add_storage_line(height, [4, 17], [4, 24])
    warehouse_3d.add_storage_line(height, [2, 3], [2, 26])

    # vertical storage line
    warehouse_3d.add_storage_line(height, [3, 2], [19, 2])
    warehouse_3d.add_storage_line(height, [5, 4], [19, 4])
    warehouse_3d.add_storage_line(height, [5, 14], [19, 14])
    warehouse_3d.add_storage_line(height, [5, 16], [19, 16])
    warehouse_3d.add_storage_line(height, [5, 25], [19, 25])
    warehouse_3d.add_storage_line(height, [3, 27], [19, 27])


    # Add objects
    warehouse_3d.add_object(2, 6, 1)
    warehouse_3d.add_object(9, 4, 0)
    warehouse_3d.add_object(4, 20, 2)

    # Add checkpoints
    warehouse_3d.add_checkpoint(20, 1, 0)
    warehouse_3d.add_checkpoint(20, 1, 1)
    warehouse_3d.add_checkpoint(20, 1, 2)

    warehouse_3d.add_checkpoint(20, 9, 0)
    warehouse_3d.add_checkpoint(20, 9, 1)
    warehouse_3d.add_checkpoint(20, 9, 2)

    warehouse_3d.add_checkpoint(10, 9, 0)
    warehouse_3d.add_checkpoint(10, 9, 1)
    warehouse_3d.add_checkpoint(10, 9, 2)

    warehouse_3d.add_checkpoint(20, 21, 0)
    warehouse_3d.add_checkpoint(20, 21, 1)
    warehouse_3d.add_checkpoint(20, 21, 2)

    warehouse_3d.add_checkpoint(10, 21, 0)
    warehouse_3d.add_checkpoint(10, 21, 1)
    warehouse_3d.add_checkpoint(10, 21, 2)

    warehouse_3d.add_checkpoint(20, 28, 0)
    warehouse_3d.add_checkpoint(20, 28, 1)
    warehouse_3d.add_checkpoint(20, 28, 2)

    warehouse_3d.add_checkpoint(1, 1, 0)
    warehouse_3d.add_checkpoint(1, 1, 1)
    warehouse_3d.add_checkpoint(1, 1, 2)

    warehouse_3d.add_checkpoint(1, 28, 0)
    warehouse_3d.add_checkpoint(1, 28, 1)
    warehouse_3d.add_checkpoint(1, 28, 2)

    # Visualisation
    warehouse_3d.display()

    # Test calcul de distance
    distance = warehouse_3d.compute_manhattan_distance((0, 0, 0), (10, 10, 0))
    print(f"Distance : {distance}")

