import pandas as pd
import numpy as np
from math import sqrt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

#################################################################################
#                        FONCTIONS DE RECONSTRUCTION DE TRAJECTOIRES
#################################################################################

# -----------------------------------------------------------------------------
#                         FONCTIONS PRINCIPALES
# -----------------------------------------------------------------------------


def ajouter_parametres_trajectoire(csv_path):
    """
    Ajoute des paramètres dynamiques à chaque trajectoire (vitesse, accélération, distance...),
    en les calculant uniquement à l'intérieur de chaque objet.
    """
    df = pd.read_csv(csv_path, delimiter=";")

    if not all(col in df.columns for col in
               ['object', 'time', 'XSplined', 'YSplined', 'ZSplined', 'VXSplined', 'VYSplined', 'VZSplined']):
        raise ValueError("Le fichier CSV doit contenir les colonnes nécessaires.")

    # Vitesse scalaire
    df['Speed'] = np.sqrt(df['VXSplined']**2 + df['VYSplined']**2 + df['VZSplined']**2)

    # Fonctions auxiliaires par objet
    def compute_features(group):
        # Tri par temps (important)
        group = group.sort_values('time')

        # Différences temporelles
        dt = group['time'].diff()

        # Accélérations
        group['AXSplined'] = group['VXSplined'].diff() / dt
        group['AYSplined'] = group['VYSplined'].diff() / dt
        group['AZSplined'] = group['VZSplined'].diff() / dt

        # Accélération scalaire
        group['Acceleration'] = np.sqrt(group['AXSplined']**2 + group['AYSplined']**2 + group['AZSplined']**2)

        # Distance parcourue
        group['DistanceTravelled'] = np.sqrt(
            group['XSplined'].diff()**2 +
            group['YSplined'].diff()**2 +
            group['ZSplined'].diff()**2
        )

        # Produit scalaire pour calcul de l’angle
        v_shift = group[['VXSplined', 'VYSplined', 'VZSplined']].shift()
        v_now = group[['VXSplined', 'VYSplined', 'VZSplined']]
        dot_product = (v_shift * v_now).sum(axis=1)

        norm_product = (v_shift.pow(2).sum(axis=1).pow(0.5)) * (v_now.pow(2).sum(axis=1).pow(0.5))
        group['AngleBetweenPoints'] = np.arccos(dot_product / norm_product.clip(lower=1e-8))

        # Accélération tangentielle
        a_now = group[['AXSplined', 'AYSplined', 'AZSplined']]
        dot_av = (a_now * v_now).sum(axis=1)
        group['TangentialAcceleration'] = dot_av / group['Speed'].replace(0, np.nan)

        # Courbure : norme de v x a / |v|^3
        cross = np.cross(v_now.fillna(0).values, a_now.fillna(0).values)
        cross_norm = np.linalg.norm(cross, axis=1)
        group['Curvature'] = cross_norm / (group['Speed']**3).replace(0, np.nan)

        return group

    # Appliquer à chaque objet
    df = df.groupby('object', group_keys=False).apply(compute_features)

    # Sauvegarde
    new_csv_path = csv_path.replace(".csv", "_avec_features.csv")
    df.to_csv(new_csv_path, sep=";", index=False)
    print(f"Fichier enrichi sauvegardé : {new_csv_path}")

    return df




def comparer_objets(data, seuil_temps, seuil_distance, debug=False):
    """
    Accepte soit un DataFrame, soit un chemin CSV.
    """
    if isinstance(data, str):
        try:
            df = pd.read_csv(data, delimiter=';')
        except Exception as e:
            raise ValueError(f"Erreur lors de la lecture du fichier CSV : {e}")
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("L'argument 'data' doit être un chemin CSV (str) ou un DataFrame.")

    # Vérifier les colonnes nécessaires
    for col in ["object", "time", "XSplined", "YSplined", "ZSplined"]:
        if col not in df.columns:
            raise ValueError(f"La colonne '{col}' est manquante dans le fichier CSV.")

    stats = df.groupby("object").agg(
        time_min=("time", "min"),
        time_max=("time", "max")
    ).reset_index()

    objets = stats["object"].tolist()
    n = len(objets)
    matrice = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            obj_i = stats.iloc[i]
            obj_j = stats.iloc[j]

            if obj_i["time_max"] < obj_j["time_min"]:
                delta_t = obj_j["time_min"] - obj_i["time_max"]
                if delta_t <= seuil_temps:
                    pos_i = df[(df["object"] == obj_i["object"]) & (df["time"] == obj_i["time_max"])][
                        ["XSplined", "YSplined", "ZSplined"]]
                    pos_j = df[(df["object"] == obj_j["object"]) & (df["time"] == obj_j["time_min"])][
                        ["XSplined", "YSplined", "ZSplined"]]

                    if not pos_i.empty and not pos_j.empty:
                        xi, yi, zi = pos_i.iloc[0]
                        xj, yj, zj = pos_j.iloc[0]
                        distance = sqrt((xj - xi)**2 + (yj - yi)**2 + (zj - zi)**2)

                        if distance <= seuil_distance:
                            matrice[i, j] = 1

    matrice_df = pd.DataFrame(matrice, index=objets, columns=objets)

    return matrice_df



def cluster_objects(df, n_clusters, debug=False):
    """
    Fonction de clustering qui prend en compte les nouvelles données (vitesse, accélération, etc.).

    :param df: DataFrame contenant les données des trajectoires (avec les nouvelles colonnes calculées).
    :param n_clusters: Le nombre de clusters à générer.
    :param debug: Si True, affiche des informations de débogage.
    :return: DataFrame avec les clusters assignés à chaque objet.
    """

    # Nouvelles colonnes à utiliser pour le clustering
    colonnes = [
        'XSplined', 'YSplined', 'ZSplined',
        'VXSplined', 'VYSplined', 'VZSplined', 'Speed',
        'AXSplined', 'AYSplined', 'AZSplined', 'Acceleration',
        'TangentialAcceleration',  'DistanceTravelled', 'Curvature'
    ]

    # Moyenne des valeurs de chaque colonne par objet
    features = df.groupby('object')[colonnes].mean()

    # Supprimer les objets avec des valeurs manquantes
    features_clean = features.dropna()

    if debug:
        n_total = features.shape[0]
        n_clean = features_clean.shape[0]
        print(f"Objets ignorés à cause de valeurs manquantes : {n_total - n_clean} / {n_total}")

    # Appliquer le KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    features_clean['cluster'] = kmeans.fit_predict(features_clean)

    if debug:
        print(" Résumé des clusters trouvés :")
        print(features_clean['cluster'].value_counts())
        print(features_clean[['cluster']])

    return features_clean



def extraire_chaines_connexes(matrice, clusters, debug=False):

    objets_par_cluster = clusters.groupby("cluster").groups

    chaines = []

    for cluster_id, objets_cluster in objets_par_cluster.items():
        objets_cluster = list(objets_cluster)
        sous_matrice = matrice.loc[objets_cluster, objets_cluster]

        if debug:
            print(f" Cluster {cluster_id} → objets : {objets_cluster}")
            print("Sous-matrice :")
            print(sous_matrice)

        visited = set()

        def dfs(obj, chaine):
            visited.add(obj)
            chaine.append(obj)
            for neighbor in sous_matrice.columns[sous_matrice.loc[obj] == 1]:
                if neighbor not in visited:
                    dfs(neighbor, chaine)

        for obj in objets_cluster:
            if obj not in visited:
                chaine = []
                dfs(obj, chaine)
                if len(chaine) > 1:
                    if debug:
                        print(f"Chaîne détectée : {chaine}")
                    chaines.append({'cluster': cluster_id, 'objects': chaine})

    return pd.DataFrame(chaines)

# -----------------------------------------------------------------------------
#                         FONCTIONS Graphiques
# -----------------------------------------------------------------------------

def plot_heatmap(matrice_df, title="Matrice de connexions"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrice_df, cmap="YlOrRd", cbar=True, square=True, linewidths=0.5, linecolor='gray')
    plt.title(title)
    plt.xlabel("Objet cible")
    plt.ylabel("Objet source")
    plt.tight_layout()
    plt.savefig("heatmap_matrice.png")
    plt.show()

def plot_reconstitute(csv_file, output_file="reconstitution_graphique.png"):
    """
    Lit un fichier CSV, trace 'object' en fonction de 'time' et sauvegarde le graphique.

    :param csv_file: Chemin vers le fichier CSV
    :param output_file: Nom du fichier de sortie pour sauvegarder le graphique (PNG par défaut)
    """
    # Lecture du fichier CSV
    df = pd.read_csv(csv_file, sep=";")

    # Vérification des colonnes nécessaires
    required_cols = {'time', 'object', 'oldobject'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Le fichier CSV doit contenir les colonnes {required_cols}.")

    # Conversion time si besoin
    if not pd.api.types.is_numeric_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])

    # Tri par temps
    df = df.sort_values(by='time')

    # Création du graphique
    plt.figure(figsize=(14, 7))

    # Une couleur par objet
    color_map = plt.cm.get_cmap('tab20', df['object'].nunique())
    object_colors = {obj: color_map(i) for i, obj in enumerate(df['object'].unique())}

    # Tracer les segments
    for obj, obj_group in df.groupby('object'):
        obj_group = obj_group.sort_values(by='time')

        # Identifier les changements de oldobject
        obj_group['segment'] = (obj_group['oldobject'] != obj_group['oldobject'].shift()).cumsum()

        for _, segment in obj_group.groupby('segment'):
            if len(segment) > 1:
                plt.plot(
                    segment['time'], segment['object'],
                    color=object_colors[obj],
                    linewidth=0.5
                )
                # Ajouter un point/crochet à la fin
                plt.plot(
                    segment['time'].iloc[-1], segment['object'].iloc[-1],
                    marker='o', markersize=4,
                    color=object_colors[obj]
                )

    plt.xlabel('Time')
    plt.ylabel('Object')
    plt.title('Reconstitution avec fin de segments marquée')
    plt.grid(True)
    plt.tight_layout()

    # Sauvegarde et affichage
    plt.savefig(output_file, dpi=300)
    print(f"Graphique sauvegardé sous : {os.path.abspath(output_file)}")



def plot_transition_histograms(csv_file, time_output="histogram_time.png", distance_output="histogram_distance.png"):
    """
    Calcule les temps et les distances entre les transitions de oldobject pour un même object,
    puis trace deux histogrammes : un pour les temps, un pour les distances.

    :param csv_file: Chemin vers le fichier CSV
    :param time_output: Nom du fichier pour l'histogramme des temps
    :param distance_output: Nom du fichier pour l'histogramme des distances
    """
    df = pd.read_csv(csv_file, sep=";")

    # Vérification des colonnes nécessaires
    required_cols = {'time', 'object', 'oldobject', 'XSplined', 'YSplined', 'ZSplined'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Le fichier CSV doit contenir les colonnes {required_cols}.")

    # Conversion temps si nécessaire
    if not pd.api.types.is_numeric_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])

    df = df.sort_values(by=['object', 'time'])

    transition_durations = []
    transition_distances = []

    # Traitement par object
    for obj, obj_group in df.groupby('object'):
        obj_group = obj_group.copy()
        obj_group['segment'] = (obj_group['oldobject'] != obj_group['oldobject'].shift()).cumsum()

        segments = list(obj_group.groupby('segment'))

        for i in range(len(segments) - 1):
            current_seg = segments[i][1]
            next_seg = segments[i + 1][1]

            end_time = current_seg['time'].iloc[-1]
            start_time = next_seg['time'].iloc[0]

            # Temps écoulé
            delta = (start_time - end_time).total_seconds() if hasattr(end_time, 'total_seconds') else start_time - end_time
            if delta >= 0:
                transition_durations.append(delta)

                # Distance entre fin du segment et début du suivant
                x1, y1, z1 = current_seg[['XSplined', 'YSplined', 'ZSplined']].iloc[-1]
                x2, y2, z2 = next_seg[['XSplined', 'YSplined', 'ZSplined']].iloc[0]
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
                transition_distances.append(dist)

    # Histogramme des temps
    plt.figure(figsize=(10, 4))
    plt.hist(transition_durations, bins=30, color='steelblue', edgecolor='black')
    plt.xlabel('Temps écoulé (secondes)')
    plt.ylabel('Nombre de transitions')
    plt.title('Histogramme des temps entre transitions de oldobject')
    plt.tight_layout()
    plt.savefig(time_output, dpi=300)
    print(f"Histogramme des temps sauvegardé sous : {os.path.abspath(time_output)}")


    # Histogramme des distances
    plt.figure(figsize=(10, 4))
    plt.hist(transition_distances, bins=30, color='darkorange', edgecolor='black')
    plt.xlabel('Distance 3D entre fin et début de oldobject')
    plt.ylabel('Nombre de transitions')
    plt.title('Histogramme des distances entre segments')
    plt.tight_layout()
    plt.savefig(distance_output, dpi=300)
    print(f"Histogramme des distances sauvegardé sous : {os.path.abspath(distance_output)}")

def plot_mirrored_duration_histogram(csv_file, output_file="mirrored_duration_histogram.png"):
    """
    Crée un histogramme en miroir comparant les durées totales de 'object' et 'oldobject'.

    :param csv_file: Chemin vers le fichier CSV
    :param output_file: Nom du fichier image de sortie
    """
    df = pd.read_csv(csv_file, sep=";")

    # Vérification des colonnes nécessaires
    required_cols = {'time', 'object', 'oldobject'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Le fichier CSV doit contenir les colonnes {required_cols}.")

    # Supposé: df['time'] contient déjà des secondes (float ou int)
    df = df.sort_values(by='time')

    # Durée totale pour chaque groupement
    object_durations = df.groupby('object')['time'].agg(lambda x: x.max() - x.min())
    oldobject_durations = df.groupby('oldobject')['time'].agg(lambda x: x.max() - x.min())

    # Bins partagés
    all_durations = pd.concat([object_durations, oldobject_durations])
    bins = np.histogram_bin_edges(all_durations, bins=30)

    # Histogrammes
    object_counts, _ = np.histogram(object_durations, bins=bins)
    oldobject_counts, _ = np.histogram(oldobject_durations, bins=bins)

    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    width = np.diff(bins)

    # Tracé
    plt.figure(figsize=(12, 6))
    plt.bar(bin_centers, object_counts, width=width, color='mediumseagreen', label='Object', edgecolor='black')
    plt.bar(bin_centers, -oldobject_counts, width=width, color='mediumpurple', label='Oldobject', edgecolor='black')

    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel('Durée totale (secondes)')
    plt.ylabel('Nombre d\'éléments')
    plt.title('Histogramme miroir des durées totales : Object (haut) vs Oldobject (bas)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Histogramme miroir sauvegardé sous : {os.path.abspath(output_file)}")


# -----------------------------------------------------------------------------
#                         FONCTIONS SUPPRIMER LES DOUBLONS
# -----------------------------------------------------------------------------

def calculer_distance_spatiale(df, id1, id2):
    p1 = df[df['object'] == id1][['XSplined', 'YSplined']].mean()
    p2 = df[df['object'] == id2][['XSplined', 'YSplined']].mean()
    # Calcul de la distance Euclidienne entre les deux objets
    distance = np.sqrt((p1['XSplined'] - p2['XSplined']) ** 2 + (p1['YSplined'] - p2['YSplined']) ** 2)

    return distance


def calculer_ressemblance_intra_cluster(features_df, id1, id2):
    cols = ['XSplined', 'YSplined', 'ZSplined', 'VXSplined', 'VYSplined', 'VZSplined']
    v1 = features_df.loc[id1, cols]
    v2 = features_df.loc[id2, cols]
    return np.linalg.norm(v1 - v2)


def score_lien(row, temps_min, features_obj, connexions_df,
               poids_temps, poids_distance, poids_ressemblance, bonus_cible_source):
    source, cible = row['objet_source'], row['objet_cible']

    # Similarité temporelle (diff absolue)
    delta_t = abs(temps_min[cible] - temps_min[source])

    # Similarité spatiale (distance moyenne XY)
    dist_spatiale = ((features_obj.loc[source] - features_obj.loc[cible]) ** 2)[['XSplined', 'YSplined']].sum() ** 0.5

    # Similarité globale sur les caractéristiques
    dist_caracs = ((features_obj.loc[source] - features_obj.loc[cible]) ** 2).sum() ** 0.5

    # Bonus si la cible est aussi une source
    bonus = bonus_cible_source if cible in connexions_df['objet_source'].values else 0

    return (
        poids_temps * delta_t +
        poids_distance * dist_spatiale +
        poids_ressemblance * dist_caracs -
        bonus
    )

def garder_connexion_optimale(connexions, df, features=None,
                              poids_temps=1.0,
                              poids_distance=1.0,
                              poids_ressemblance=1.0,
                              bonus_cible_source=0.0):
    # Gérer le cas sans cluster
    if features is None:
        # Utiliser toutes les colonnes numériques moyennées par objet
        colonnes_valables = df.select_dtypes(include=[np.number]).columns.tolist()
        colonnes_valables = [col for col in colonnes_valables if col != "time"]
        features = df.groupby("object")[colonnes_valables].mean()

    # Cas avec ou sans cluster (si elle existe)
    if 'cluster' in features.columns:
        connexions_df = pd.DataFrame(connexions, columns=["objet_source", "objet_cible", "cluster"])
        features_obj = features.drop(columns=['cluster'])
    else:
        connexions_df = pd.DataFrame(connexions, columns=["objet_source", "objet_cible"])
        features_obj = features

    temps_min = df.groupby('object')['time'].min().to_dict()

    connexions_df['score'] = connexions_df.apply(
        lambda row: score_lien(
            row,
            temps_min,
            features_obj,
            connexions_df,
            poids_temps,
            poids_distance,
            poids_ressemblance,
            bonus_cible_source
        ),
        axis=1
    )

    # Trier pour garder la meilleure connexion par source
    connexions_df = connexions_df.sort_values(by='score')

    sources_utilisées = set()
    cibles_utilisées = set()
    connexions_finales = []

    for _, row in connexions_df.iterrows():
        source = row['objet_source']
        cible = row['objet_cible']
        cluster = row['cluster'] if 'cluster' in row else -1  # ou None

        # Un seul lien sortant par source et un seul entrant par cible
        if source not in sources_utilisées and cible not in cibles_utilisées:
            connexions_finales.append((source, cible, cluster))
            sources_utilisées.add(source)
            cibles_utilisées.add(cible)

    connexions_finales_df = pd.DataFrame(
        connexions_finales,
        columns=["objet_source", "objet_cible", "cluster"]
    ).astype(int)

    connexions_finales_df = connexions_finales_df.sort_values(by="objet_source")

    return connexions_finales_df


def reconstituer_trajectoires(df_connexions):
    # Construction d'un graphe simple
    suivants = dict(zip(df_connexions["objet_source"], df_connexions["objet_cible"]))
    precedents = {v: k for k, v in suivants.items()}

    # Identifier les têtes de trajectoires (objets qui ne sont pas cibles)
    têtes = [source for source in suivants.keys() if source not in precedents]

    trajectoires = []

    for tête in sorted(têtes):
        traj = [tête]
        courant = tête

        while courant in suivants:
            courant = suivants[courant]
            traj.append(courant)

        objet_source = traj[0]
        objets_cibles = traj[1:]  # on exclut le premier
        trajectoires.append((objet_source, ",".join(str(int(o)) for o in objets_cibles)))

    return pd.DataFrame(trajectoires, columns=["objet_source", "objet_cible"])

# -----------------------------------------------------------------------------
#                         FONCTIONS OBTENIR FIHCIERS DE SORTIES
# -----------------------------------------------------------------------------

def update_target_with_source(connexions_df, df, seuil_duree):
    """
    Met à jour les objets cibles dans le DataFrame des trajectoires en remplaçant l'objet cible par l'objet source,
    tout en ajoutant une colonne 'oldobject' pour conserver l'objet cible d'origine.
    :param connexions_df: DataFrame des connexions avec les colonnes 'objet_source' et 'objet_cible'
    :param df: DataFrame des trajectoires
    :param seuil_duree: Seuil de durée minimale pour conserver les trajectoires (en temps)
    :return: DataFrame des trajectoires mis à jour
    """
    # Créer une copie du DataFrame des trajectoires pour ne pas modifier l'original
    df_copy = df.copy()

    # Ajouter une nouvelle colonne 'oldobject' qui sera initialisée avec les valeurs d'origine de la colonne 'object'
    df_copy['oldobject'] = df_copy['object']

    # Créer un dictionnaire des objets source avec leurs cibles
    connexions_dict = {}
    for _, row in connexions_df.iterrows():
        source = row['objet_source']
        cibles = str(row['objet_cible']).split(",")
        connexions_dict[source] = [int(c) for c in cibles]

    # Remplacer les objets cibles par l'objet source dans la colonne 'object'
    for source, cibles in connexions_dict.items():
        for cible in cibles:
            # Mise à jour des lignes où 'object' est égal à 'cible'
            df_copy.loc[df_copy['object'] == cible, 'object'] = source

    # Réorganiser les colonnes pour que 'oldobject' soit la deuxième colonne
    cols = ['object', 'oldobject'] + [col for col in df_copy.columns if col not in ['object', 'oldobject']]
    df_copy = df_copy[cols]

    # Filtrer les objets dont la durée de trajectoire est inférieure au seuil
    objets_a_conserver = []
    for obj in df_copy['object'].unique():
        # Calculer la durée de la trajectoire pour cet objet
        time_min = df_copy[df_copy['object'] == obj]['time'].min()
        time_max = df_copy[df_copy['object'] == obj]['time'].max()
        duree_trajectoire = time_max - time_min

        # Si la durée est supérieure ou égale au seuil, on garde les lignes
        if duree_trajectoire >= seuil_duree:
            objets_a_conserver.append(obj)

    # Garder uniquement les lignes des objets dont la durée est supérieure ou égale au seuil
    df_copy = df_copy[df_copy['object'].isin(objets_a_conserver)]

    return df_copy


def save_reconstituted_file(input_file, connexions_df, df, seuil_duree):
    """
    Sauvegarde le fichier des trajectoires après mise à jour des objets cibles.
    Le fichier est sauvegardé avec le même nom que le fichier d'entrée, mais avec '_reconstitue' ajouté à la fin.
    :param input_file: Le fichier d'entrée
    :param connexions_df: DataFrame des connexions entre objets
    :param df: DataFrame des trajectoires à mettre à jour
    :param seuil_duree: Seuil de durée minimale pour conserver les trajectoires (en temps)
    """
    # Appliquer la mise à jour des cibles dans les trajectoires
    df_modifie = update_target_with_source(connexions_df, df, seuil_duree)

    # Ne conserver que les colonnes souhaitées
    colonnes_a_conserver = [
        'object', 'oldobject', 'time',
        'XSplined', 'YSplined', 'ZSplined',
        'VXSplined', 'VYSplined', 'VZSplined'
    ]
    df_modifie = df_modifie[colonnes_a_conserver]

    # Générer le nom du fichier de sortie avec le suffixe '_reconstitue'
    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}_reconstitue{ext}"

    # Sauvegarder le fichier modifié
    df_modifie.to_csv(output_file, sep=';', index=False)
    print(f"Le fichier des trajectoires a été mis à jour et sauvegardé sous : {output_file}")

