# Drone Planification Project

## adjacency_matrix.py

Tha adjacency matrix is generated from the warehouse matrix. The value at position (i, j) of the warehouse matrix correspond to the category of the case (i, j) in the warehouse, as defined in the dictionnary category_mapping.

The script adjacency_matrix.py goes through the warehouse matrix first on the z axis, then x axis, then y axis.

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

![Texte alternatif](drone-planification-project/Data_test/adjmatrix_schema.png "Adjacency matrix for one_level_U_matrix").

## Name
Trafic de drones dans un espace restreint.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Usage
TODO : manuel d'utilisation pour un employé de l'entrepôt pour avoir la planification chaque matin.

## Project status
Work in progress.
