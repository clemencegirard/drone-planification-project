import numpy as np
import matplotlib.pyplot as plt

# Dimensions de la grille
rows, cols = 30, 30

# Créer une matrice représentant l'entrepôt
# 0: espace libre, 1: obstacle
warehouse = np.zeros((rows, cols))
warehouse[3, 4] = 1  # Ajouter un obstacle
warehouse[5, 7] = 1

def creation_rectanglle(angle_no : tuple, angle_ne : tuple, angle_se : tuple, angle_so: tuple):

    if abs(angle_no[1]-angle_ne[1]) == abs(angle_so[1]-angle_se[1]):
        largeur = abs(angle_no[1]-angle_ne[1])
    else :
        return 'Pas un rectangle'
    if abs(angle_no[0]-angle_so[0]) == abs(angle_ne[0]-angle_se[0]):
        hauteur = abs(angle_no[0]-angle_so[0])

    for i in range(hauteur):
        for j in range(largeur):
            

    return warehouse



# Affichage
plt.figure(figsize=(6, 6))
plt.imshow(warehouse, cmap="Greys", origin="upper")
plt.grid(which="both", color="black", linewidth=0.4)
plt.xticks(range(cols))
plt.yticks(range(rows))
plt.show()
