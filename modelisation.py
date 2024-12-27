import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray


class Entrepot():
    def __init__(self, rows : int, cols:int):
        self.rows = rows
        self.cols = cols
        self.mat = np.zeros((self.rows, self.cols))

    def display(self):
        plt.figure(figsize=(6, 6))
        plt.imshow(self.mat, cmap="Greys", origin="upper")
        plt.grid(which="both", color="black", linewidth=0.4)
        plt.xticks(range(self.cols))
        plt.yticks(range(self.rows))
        plt.show()

    def creation_rectangle(self,angle_no : tuple, angle_ne : tuple, angle_se : tuple, angle_so: tuple) -> ndarray:

        if abs(angle_no[1]-angle_ne[1]) == abs(angle_so[1]-angle_se[1]):
            largeur = abs(angle_no[1]-angle_ne[1])
        else :
            return 'Pas un rectangle'
        if abs(angle_no[0]-angle_so[0]) == abs(angle_ne[0]-angle_se[0]):
            hauteur = abs(angle_no[0]-angle_so[0])
        else :
            return 'Pas un rectangle'
        for i in range(hauteur):
            for j in range(largeur):
                self.mat[i+angle_no[0],j+angle_no[1]] = 1

        return self.mat


entrepot_exemple = Entrepot(30,30)
entrepot_exemple.creation_rectangle([10,10], [10,20], [20,10], [20,20])
entrepot_exemple.display()

