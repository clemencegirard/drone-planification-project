import numpy as np
import matplotlib.pyplot as plt


class Entrepot3D:
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

            # Affichage des cercles orange pour les cases où self.mat == 2 (emplacement objet)
            # et des cercles rouges pour les cases où self.mat == 3 (objet)
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

        # Ajout de la légende
        labels = [
            "Zone libre",
            "Étagère",
            "Emplacement objet",
            "Présence d'un Objet",
            "Point de passage"
        ]
        colors = ["white", "gray", "orange", "red", "blue"]

        # Création des handles pour la légende
        handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10) for c in colors]
        fig.legend(handles, labels, loc="upper right", fontsize=10)

        plt.show()

    def print_mat(self):
        return print(self.mat)

    def add_etagere(self, hauteur: int, angle_no: tuple, angle_ne: tuple, angle_se: tuple, angle_so: tuple):
        if not (0 <= hauteur <= self.height):
            return print(f"Niveau {hauteur} hors limites")

        # Vérification des dimensions du rectangle
        if abs(angle_no[1] - angle_ne[1]) == abs(angle_so[1] - angle_se[1]):
            largeur = abs(angle_no[1] - angle_ne[1])
        else:
            return print("Pas un rectangle")
        if abs(angle_no[0] - angle_so[0]) == abs(angle_ne[0] - angle_se[0]):
            hauteur_rectangle = abs(angle_no[0] - angle_so[0])
        else:
            return print("Pas un rectangle")

        # Remplissage des niveaux inférieurs
        for h in range(hauteur):  # Inclut la couche spécifiée et toutes celles en dessous
            for i in range(hauteur_rectangle):
                for j in range(largeur):
                    self.mat[i + angle_no[0], j + angle_no[1], h] = 1

    def add_file_rangement(self, hauteur: int, c1: tuple, c2: tuple):
        if not (0 <= hauteur <= self.height):  # Vérification des limites
            return print(f"Niveau {hauteur} hors limites")

        x1, y1 = c1
        x2, y2 = c2
        isArmoire = False

        if x1 == x2:  # Ligne verticale
            for h in range(hauteur):  # Remplir tous les niveaux jusqu'à hauteur inclus
                for j in range(abs(y2 - y1) + 1):
                    if self.mat[x1, min(y1, y2) + j, h] == 1 or self.mat[x1, min(y1, y2) + j, h] == 2:
                        isArmoire = True
                    else:
                        return print("La file précisée est hors armoire")
                if isArmoire:
                    for j in range(abs(y2 - y1) + 1):
                        self.mat[x1, min(y1, y2) + j, h] = 2

        elif y1 == y2:  # Ligne horizontale
            for h in range(hauteur):  # Remplir tous les niveaux jusqu'à hauteur inclus
                for j in range(abs(x2 - x1) + 1):
                    if self.mat[min(x1, x2) + j, y1, h] == 1 or self.mat[min(x1, x2) + j, y1, h] == 2:
                        isArmoire = True
                    else:
                        return print("La file précisée est hors armoire")
                if isArmoire:
                    for j in range(abs(x2 - x1) + 1):
                        self.mat[min(x1, x2) + j, y1, h] = 2

        else:
            return print("Ce n'est pas une file")

    def add_object(self, rows: int, col: int, level: int):

        if self.mat[rows, col, level] == 2 :
            self.mat[rows, col, level] = 3
        else:
            print("Il n'y a pas d'étagère ici")

    def add_checkpoint(self, rows: int, col: int, level: int):

        if self.mat[rows, col, level] == 0 :
            self.mat[rows, col, level] = 4
        else:
            print("Il y a un obstacle ici")



if __name__ == "__main__":

    hauteur = 3
    # Exemple d'utilisation
    entrepot_3d = Entrepot3D(30, 30, hauteur)

    # Ajout d'étagères au niveau 0

    #étagères
    entrepot_3d.add_etagere(hauteur,[2,2], [2,28], [5,2], [5,28])
    entrepot_3d.add_etagere(hauteur,[2,2], [2,5], [20,5], [20,2])
    entrepot_3d.add_etagere(hauteur,[2,14], [2,17], [20,17], [20,14])
    entrepot_3d.add_etagere(hauteur,[2,25], [2,28], [20,28], [20,25])

    # file rangement horizontal
    entrepot_3d.add_file_rangement(hauteur, [4, 5], [4, 13])
    entrepot_3d.add_file_rangement(hauteur, [4, 17], [4, 24])
    entrepot_3d.add_file_rangement(hauteur,[2, 3], [2, 26])

    # file rangement vertical
    entrepot_3d.add_file_rangement(hauteur,[3, 2], [19, 2])
    entrepot_3d.add_file_rangement(hauteur,[5, 4], [19, 4])
    entrepot_3d.add_file_rangement(hauteur,[5, 14], [19, 14])
    entrepot_3d.add_file_rangement(hauteur,[5, 16], [19, 16])
    entrepot_3d.add_file_rangement(hauteur,[5, 25], [19, 25])
    entrepot_3d.add_file_rangement(hauteur,[3, 27], [19, 27])


    #ajout d'objet
    entrepot_3d.add_object(2,6,1)
    entrepot_3d.add_object(9,4,0)
    entrepot_3d.add_object(4,20,2)

    # ajout point de passage
    entrepot_3d.add_checkpoint(20, 1, 0)
    entrepot_3d.add_checkpoint(20, 1, 1)
    entrepot_3d.add_checkpoint(20, 1, 2)

    entrepot_3d.add_checkpoint(20, 9, 0)
    entrepot_3d.add_checkpoint(20, 9, 1)
    entrepot_3d.add_checkpoint(20, 9, 2)

    entrepot_3d.add_checkpoint(20, 21, 0)
    entrepot_3d.add_checkpoint(20, 21, 1)
    entrepot_3d.add_checkpoint(20, 21, 2)

    entrepot_3d.add_checkpoint(20, 28, 0)
    entrepot_3d.add_checkpoint(20, 28, 1)
    entrepot_3d.add_checkpoint(20, 28, 2)

    entrepot_3d.add_checkpoint(1, 1, 0)
    entrepot_3d.add_checkpoint(1, 1, 1)
    entrepot_3d.add_checkpoint(1, 1, 2)

    entrepot_3d.add_checkpoint(1, 28, 0)
    entrepot_3d.add_checkpoint(1, 28, 1)
    entrepot_3d.add_checkpoint(1, 28, 2)


    # Visualisation
    entrepot_3d.display()
