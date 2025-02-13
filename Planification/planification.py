import csv
import os
import pandas as pd
import numpy as np


def open_csv(csv_file_name: str):
    """Ouvre un fichier CSV et retourne un DataFrame pandas."""
    path = os.path.join(os.path.dirname(__file__), "TaskList", csv_file_name)

    # Lecture du fichier CSV avec pandas
    return pd.read_csv(path, header=0)  # header=0 pour s'assurer que la première ligne est l'en-tête


def assigner_drones(csv_file, nb_drone):
    """Attribue un drone à chaque tâche de manière cyclique."""
    csv_file['drone'] = ""  # Initialise la colonne

    compteur = 0
    for index, __ in csv_file.iterrows():
        csv_file.loc[index, 'drone'] = f'd{compteur % nb_drone + 1}'
        compteur += 1

    return csv_file


def calcul_distance(matrice_adjacence : np.ndarray, dictionnaire_position : dict, t1 : tuple ,t2 : tuple):

    n1 = dictionnaire_position[t1]
    n2 = dictionnaire_position[t2]

    return matrice_adjacence[n1][n2]



def preparation_donnees(matrice_adjacence,dictionnaire_position, csv_file_name, nb_drone: int):
    # Générer les heures de 8h à 18h (exclu) avec un pas de 1 heure
    hour = np.arange(8, 19, 1)

    # Créer un DataFrame avec ces heures formatées
    df = pd.DataFrame(index=pd.to_datetime(hour, format='%H').strftime('%H:%M'))
    df['tache'] = ""
    df['drone'] = ""

    # Ouvrir et trier le fichier CSV
    csv_file = open_csv(csv_file_name)

    # Convertir la colonne 'time' en datetime pour un tri correct
    csv_file['time'] = pd.to_datetime(csv_file['time'], format='%H:%M').dt.strftime('%H:%M')

    # Trier les valeurs par 'time'
    csv_file = csv_file.sort_values(by=['time'])

    # Assigner les drones
    csv_file = assigner_drones(csv_file, nb_drone)

    grouped_tasks = csv_file.groupby('time')[['id', 'drone']].agg(lambda x: ', '.join(x))

    # Mapper les tâches vers df
    df['tache'] = df.index.map(grouped_tasks['id']).fillna("")
    df['drone'] = df.index.map(grouped_tasks['drone']).fillna("")

    csv_file['task_time'] = csv_file.apply(
        lambda row: calcul_distance(
            matrice_adjacence,
            dictionnaire_position,
            [row['row0'], row['col0'], row['height0']],
            [row['row1'], row['col1'], row['height1']]
        ),
        axis=1)


    return df,csv_file




def planification(matrice_adjacence,dictionnaire_position, csv_file_name, nb_drone: int):

    """Crée un planning horaire et charge les tâches depuis un CSV."""

    df, csv_file =  preparation_donnees(matrice_adjacence,dictionnaire_position, csv_file_name, nb_drone)

    return csv_file, df



# Exécution de la fonction avec un fichier de test
csv_file, df = planification("test_planning.csv",3)

# Affichage des résultats
print("Données du fichier CSV triées :\n", csv_file)
print("Données du fichier df :\n", df)
