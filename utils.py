import pandas as pd
import numpy as np
from math import sqrt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
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
    print(f"✅ Fichier enrichi sauvegardé : {new_csv_path}")

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

    if debug:
        print("📊 Matrice spatio-temporelle :")
        print(matrice_df)

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
        print(f"\n📉 Objets ignorés à cause de valeurs manquantes : {n_total - n_clean} / {n_total}")

    # Appliquer le KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    features_clean['cluster'] = kmeans.fit_predict(features_clean)

    if debug:
        print("\n🧠 Résumé des clusters trouvés :")
        print(features_clean['cluster'].value_counts())
        print(features_clean[['cluster']])

    return features_clean



def extraire_chaines_connexes(matrice, clusters, debug=False):

    objets = matrice.index.tolist()
    objets_par_cluster = clusters.groupby("cluster").groups

    chaines = []

    for cluster_id, objets_cluster in objets_par_cluster.items():
        objets_cluster = list(objets_cluster)
        sous_matrice = matrice.loc[objets_cluster, objets_cluster]

        if debug:
            print(f"\n🔍 Cluster {cluster_id} → objets : {objets_cluster}")
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
                        print(f"🔗 Chaîne détectée : {chaine}")
                    chaines.append({'cluster': cluster_id, 'objects': chaine})

    return pd.DataFrame(chaines)

# -----------------------------------------------------------------------------
#                         FONCTIONS DEBUG
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


# -----------------------------------------------------------------------------
#                         FONCTIONS AUTO-ANALYSE
# -----------------------------------------------------------------------------

def auto_analyse(csv_path, seuils_temps, seuils_distance, n_clusters, debug=False):
    print("🔍 Lancement du mode auto-analyse...")

    resultats = []

    # Charger et enrichir les données une seule fois
    try:
        df = ajouter_parametres_trajectoire(csv_path)
    except Exception as e:
        print(f"❌ Erreur lors de l'enrichissement des données : {e}")
        return

    for t, d in zip(seuils_temps, seuils_distance):
        for n in n_clusters:

            try:
                # 1. Calculer la matrice spatio-temporelle
                matrice_df = comparer_objets(df, seuil_temps=t, seuil_distance=d, debug=debug)

                if not isinstance(matrice_df, pd.DataFrame):
                    raise ValueError(f"La matrice n'est pas un DataFrame : {type(matrice_df)}")

                # 2. Clustering des objets
                features = cluster_objects(df, n_clusters=n, debug=debug)

                if not isinstance(features, pd.DataFrame):
                    raise ValueError(f"Le résultat du clustering n'est pas un DataFrame : {type(features)}")

                # 3. Créer les connexions
                objets_connectes = []
                for obj1 in matrice_df.index:
                    for obj2 in matrice_df.columns:
                        if matrice_df.loc[obj1, obj2] == 1:
                            if obj1 in features.index and obj2 in features.index:
                                if features.loc[obj1, 'cluster'] == features.loc[obj2, 'cluster']:
                                    objets_connectes.append((obj1, obj2, int(features.loc[obj1, 'cluster'])))

                # 4. Calcul du rapport
                objets_source = [x[0] for x in objets_connectes]
                objets_cible = [x[1] for x in objets_connectes]

                nb_repetitions_source = sum([objets_source.count(obj) > 1 for obj in objets_source])
                nb_repetitions_cible = sum([objets_cible.count(obj) > 1 for obj in objets_cible])
                nb_liens = len(objets_connectes)

                rapport = (nb_liens - (nb_repetitions_source + nb_repetitions_cible)) / nb_liens if nb_liens > 0 else 0

                resultats.append({
                    'seuil_temps': t,
                    'seuil_distance': d,
                    'n_clusters': n,
                    'nb_liens': nb_liens,
                    'repetitions_source': nb_repetitions_source,
                    'repetitions_cible': nb_repetitions_cible,
                    'rapport': rapport,
                    'score_final': rapport * np.log1p(nb_liens)  # log1p = log(1 + x) pour éviter log(0)
                })


            except Exception as e:
                print(f"❌ Erreur pour t={t}, d={d}, n={n}: {e}")
                continue

    # Afficher les résultats
    if resultats:
        resultats_df = pd.DataFrame(resultats)
        resultats_df = resultats_df.sort_values(
            by=['score_final', 'seuil_temps', 'n_clusters'],
            ascending=[False, True, False]  # Max score, min temps, min cluster
        )


        # Rappel des meilleurs paramètres
        meilleur = resultats_df.iloc[0]
        print("\n✅ Meilleurs paramètres sélectionnés par l'auto-analyse :")
        print(f"   • Seuil temporel     : {meilleur['seuil_temps']}")
        print(f"   • Seuil spatial      : {meilleur['seuil_distance']}")
        print(f"   • Nombre de clusters : {int(meilleur['n_clusters'])}")
        print(f"   • Liens valides      : {meilleur['nb_liens']}")
        print(f"   • Répétitions source : {meilleur['repetitions_source']}")
        print(f"   • Répétitions cible  : {meilleur['repetitions_cible']}")
        print(f"   • Rapport final      : {round(meilleur['rapport'], 3)}")
        print(f"   • Score final        : {round(meilleur['score_final'], 3)}")

        # Recalcul final
        matrice_resultat = comparer_objets(df, seuil_temps=meilleur['seuil_temps'], seuil_distance=meilleur['seuil_distance'], debug=debug)
        if debug:
            matrice_resultat.to_csv("matrice_spatiotemporelle.csv")
            print("✅ Matrice spatio-temporelle sauvegardée.")

        features_final = cluster_objects(df, n_clusters=int(meilleur['n_clusters']), debug=debug)

        objets_connectes = []
        for obj1 in matrice_resultat.index:
            for obj2 in matrice_resultat.columns:
                if matrice_resultat.loc[obj1, obj2] == 1:
                    if obj1 in features_final.index and obj2 in features_final.index:
                        if features_final.loc[obj1, 'cluster'] == features_final.loc[obj2, 'cluster']:
                            objets_connectes.append((obj1, obj2, int(features_final.loc[obj1, 'cluster'])))

        pd.DataFrame(objets_connectes, columns=["objet_source", "objet_cible", "cluster"]).to_csv("connexions_valides.csv", index=False)
        print("✅ Connexions valides sauvegardées.")

    else:
        print("⚠️ Aucun résultat d'auto-analyse valide.")


    return pd.DataFrame(objets_connectes, columns=["objet_source", "objet_cible", "cluster"])



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

    # Générer le nom du fichier de sortie avec le suffixe '_reconstitue'
    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}_reconstitue{ext}"

    # Sauvegarder le fichier modifié
    df_modifie.to_csv(output_file, sep=';', index=False)
    print(f"✅ Le fichier des trajectoires a été mis à jour et sauvegardé sous : {output_file}")
