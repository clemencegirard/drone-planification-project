import csv
import os
import pandas as pd
import numpy as np


def open_csv(csv_file_name: str):
    # Déterminer le chemin correct
    path = os.path.join(os.path.dirname(__file__), "TaskList", csv_file_name)

    with open(path, newline='') as csv_file:
        read = csv.reader(csv_file)
        return list(read)  # Lire les données et les retourner sous forme de liste


def planification(csv_file_name):
    # Générer les heures de 8h à 18h (exclu) avec un pas de 1 heure
    hour = np.arange(8, 19, 1)

    # Créer un DataFrame avec ces heures formatées
    df = pd.DataFrame(index=pd.to_datetime(hour, format='%H').strftime('%H:%M'))

    df['tache'] = ""

    csv_file = open_csv(csv_file_name)  # Ouvre le fichier CSV

    for row in csv_file:


    return csv_file, df  # Retourne le CSV et le DataFrame avec les heures en index


csv_file, df = planification("TL_three_level_line_warehouse.csv")
print(df)

print()
