import pandas as pd
import numpy as np

np.random.seed(42)

# Paramètres généraux
n_groupes = 150
fragments_par_moustique = 4
fps = 50
dt = 1 / fps
total_time = 30.0
timestamps = np.arange(0, total_time, dt)
n_timestamps = len(timestamps)
points_par_fragment = n_timestamps // fragments_par_moustique
vitesse_base = 0.5  # m/s

# Bornes spatiales
X_MIN, X_MAX = -0.25, 2
Y_MIN, Y_MAX = -1, 0.25
Z_MIN, Z_MAX = -0.5, 1

# Centre de l'essaim
centre_essaim = np.array([(X_MIN + X_MAX) / 2, (Y_MIN + Y_MAX) / 2, (Z_MIN + Z_MAX) / 2])

# Paramètre d'attraction douce vers le centre
sigma = 0.4  # Plus petite valeur = attraction plus forte

data = []

# Pour chaque moustique
for g in range(n_groupes):
    true_id = g + 1
    object_ids = [100 * (g + 1) + i for i in range(1, fragments_par_moustique + 1)]

    # Position et vitesse de départ
    pos = np.random.uniform([X_MIN, Y_MIN, Z_MIN], [X_MAX, Y_MAX, Z_MAX])
    velocity = np.random.uniform(-0.1, 0.1, size=3)
    velocity /= np.linalg.norm(velocity) + 1e-6
    velocity *= vitesse_base

    t_index = 0  # index dans le tableau de temps

    # Déterminer aléatoirement les points de transition entre fragments
    transition_points = []
    remaining_points = n_timestamps

    # Pour chaque fragment sauf le dernier
    for _ in range(fragments_par_moustique - 1):
        # Points minimum pour le fragment courant (au moins 10% des points par fragment)
        min_points = int(0.1 * points_par_fragment)
        max_points = remaining_points - (fragments_par_moustique - len(transition_points) - 1) * min_points

        # Choisir un nombre aléatoire de points pour ce fragment
        fragment_points = np.random.randint(min_points, max_points)
        transition_points.append(fragment_points)
        remaining_points -= fragment_points

    # Le dernier fragment prend tous les points restants
    transition_points.append(remaining_points)

    for frag_id, (object_id, fragment_duration) in enumerate(zip(object_ids, transition_points)):
        fragment_data = []

        for _ in range(fragment_duration):
            if t_index >= len(timestamps):
                break
            t = timestamps[t_index]

            # Bruit + perturbation
            perturbation = np.random.normal(0, 0.15, size=3)
            perturbation[2] *= 0.3

            # Attraction douce vers le centre
            distance_essaim = np.linalg.norm(pos - centre_essaim)
            attraction_factor = np.exp(-0.5 * (distance_essaim / sigma) ** 2)
            vers_centre = (centre_essaim - pos)
            vers_centre /= distance_essaim + 1e-6
            perturbation += attraction_factor * 0.05 * vers_centre

            # Mise à jour de la vitesse
            velocity += perturbation
            velocity /= np.linalg.norm(velocity) + 1e-6
            speed = vitesse_base * np.random.uniform(0.5, 1.5)
            velocity *= speed

            # Mise à jour de la position
            pos += velocity * dt
            pos = np.clip(pos, [X_MIN, Y_MIN, Z_MIN], [X_MAX, Y_MAX, Z_MAX])
            velocity += np.random.normal(0, 0.1, size=3)

            fragment_data.append([
                object_id,
                t,
                pos[0], pos[1], pos[2],
                velocity[0], velocity[1], velocity[2]
            ])

            t_index += 1

        data.extend(fragment_data)

        # Simuler le trou (gap de tracking) sauf après le dernier fragment
        if frag_id < fragments_par_moustique - 1:
            gap_frames = np.random.randint(2, 10)  # de 0.04s à 0.18s
            if t_index + gap_frames < len(timestamps):
                # Faire avancer le moustique pendant le gap
                time_gap = dt * gap_frames
                pos = pos + velocity * time_gap
                t_index += gap_frames

# DataFrame final
df = pd.DataFrame(data, columns=[
    "object", "time", "XSplined", "YSplined", "ZSplined",
    "VXSplined", "VYSplined", "VZSplined"
])
df["object"] = df["object"].astype(int)
df.sort_values(by="time", inplace=True)

# Export
df.to_csv("jeu_test_moustiques.csv", sep=';', index=False)
print("✅ Fichier 'jeu_test_moustiques_fragmentes_decentralises.csv' généré avec transitions aléatoires.")