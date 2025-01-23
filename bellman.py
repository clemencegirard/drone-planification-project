import numpy as np
import csv


def inf_mat(M : np.ndarray):
    # Replace
    for i in range(len(M)):
        for j in range(len(M[i])):
            if i != j and M[i][j] == 0:
                M[i][j] = float('inf')
    return M


def bellman_recursif(noeud_depart: int, M: list, dist: list, pred: list, iterations: int) -> list:
    # Si les itérations sont terminées, retourne la liste des distances
    if iterations == 0:
        return dist

    n = len(M)  # Nombre de nœuds dans le graphe
    modified = False  # Variable pour suivre les modifications de distance

    # Liste des sommets où l'on commence par le noeud de départ
    ordre_sommets = [noeud_depart] + [i for i in range(n) if i != noeud_depart]

    for sommet in ordre_sommets:
        for voisin in range(n):
            if M[sommet][voisin] != float('inf'):  # Il existe une arête entre les 2 sommets considérés
                if dist[sommet] + M[sommet][voisin] < dist[voisin]:
                    dist[voisin] = dist[sommet] + M[sommet][voisin]
                    pred[voisin] = sommet
                    modified = True

    # Appelle récursivement la fonction seulement si une modification a été effectuée
    if modified:
        return bellman_recursif(noeud_depart, M, dist, pred, iterations - 1)
    else:
        return dist


def minimum_distance_matrix(M):

    n = len(M)
    C = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            dist = [float('inf')] * n
            dist[i] = 0  # La distance au nœud de départ est 0
            pred = [-1] * n  # Prédécesseur de chaque nœud (pour reconstruire le chemin)
            distance = bellman_recursif(i, M, dist, pred, n - 1)
            C[i, j] = distance[j]

    return C


if __name__ == '__main__':


    ## Test de la fonction

    with open('Data_test/M.csv', newline='') as csvfile:
        time_matrix = list(csv.reader(csvfile))

    M = []

    for i in range(len(time_matrix)):  # On convertit les str en int
        ligne = []
        for j in range(len(time_matrix[i])):
            ligne.append(int(time_matrix[i][j]))
        M.append(ligne)

    M = inf_mat(M)
    response = minimum_distance_matrix(M)

    print(response)