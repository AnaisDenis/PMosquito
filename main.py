import argparse
from utils import *
import pandas as pd
import itertools
import os


if __name__ == "__main__":
    import argparse

    # Création du parser d'argument
    parser = argparse.ArgumentParser(description='Analyse des trajectoires de moustiques')

    # Ajout des arguments avec type float pour seuil_temps et seuil_distance
    parser.add_argument('csv_path', type=str, help='Chemin vers le fichier CSV des trajectoires')
    parser.add_argument('--seuil_temps', type=float, required=True, help='Seuil temporel entre les points')
    parser.add_argument('--seuil_distance', type=float, required=True, help='Seuil de distance entre les points')
    parser.add_argument('--n_clusters', type=int, required=True, help='Nombre de clusters à utiliser')
    parser.add_argument('--auto-analyse', action='store_true', help='Activer l\'auto-analyse')
    parser.add_argument('--debug', action='store_true', help='Activer le mode débogage')
    parser.add_argument('--poids-temps', type=float, default=1.0, help='Pondération de la proximité temporelle')
    parser.add_argument('--poids-distance', type=float, default=1.0, help='Pondération de la distance spatiale')
    parser.add_argument('--poids-ressemblance', type=float, default=1.0, help='Pondération de la ressemblance intra-cluster')
    parser.add_argument('--bonus-cible-source', type=float, default=0.5, help='Bonus si la cible est aussi une source')
    parser.add_argument('--time-min-reconstitute', type=float, default=0, help='Temps minimal pour que la trajctoire parraisse dans le fichier de sortie')
    # Récupération des arguments
    args = parser.parse_args()

    if args.auto_analyse:
        seuils_temps_test = [0.2, 0.5, 1]
        seuils_distance_test = [0.1, 0.5, 1]
        nombre_clusters = [3, 6, 8]
        auto_analyse_resultat = auto_analyse(args.csv_path, seuils_temps_test, seuils_distance_test, nombre_clusters, debug=args.debug)

        # Rechargement du CSV pour effectuer la reconstitution des trajectoires après l'auto-analyse
        df = pd.read_csv(args.csv_path, delimiter=';')
        df = ajouter_parametres_trajectoire(args.csv_path)  # Enrichir les données après auto-analyse

        # Récupération des connexions optimisées de l'auto-analyse
        connexions_groupées = reconstituer_trajectoires(
            auto_analyse_resultat)  # Remplacez auto_analyse_resultat par le résultat réel

        # Sauvegarde du fichier des trajectoires reconstituées avec filtrage par durée
        seuil_duree = args.time_min_reconstitute  # Exemple de seuil de durée (en temps)
        save_reconstituted_file(args.csv_path, connexions_groupées, df, seuil_duree)

    else:

        matrice_df = comparer_objets(args.csv_path, args.seuil_temps, args.seuil_distance, debug=args.debug)


        # Affichage heatmap
        if args.debug:
            print("📊 Affichage de la heatmap des connexions...")
            plot_heatmap(matrice_df)
            # Sauvegarde CSV
            matrice_df.to_csv("matrice_spatiotemporelle.csv")
            print("✅ Matrice spatio-temporelle sauvegardée dans 'matrice_spatiotemporelle.csv'")


        df = pd.read_csv(args.csv_path, delimiter=';')
        # Ajouter les nouveaux paramètres avant le clustering
        df = ajouter_parametres_trajectoire(args.csv_path)  # Enrichir les données

        features = cluster_objects(df, n_clusters=args.n_clusters, debug=args.debug)

        # Création du fichier final de connexions valides (non optimisées)
        objets_connectes = []
        for obj1 in matrice_df.index:
            for obj2 in matrice_df.columns:
                if matrice_df.loc[obj1, obj2] == 1:
                    if features.loc[obj1, 'cluster'] == features.loc[obj2, 'cluster']:
                        objets_connectes.append((obj1, obj2, int(features.loc[obj1, 'cluster'])))

        if args.debug:
        # Sauvegarde des connexions non optimisées
            pd.DataFrame(objets_connectes, columns=["objet_source", "objet_cible", "cluster"]).to_csv(
                "connexions_non_optimisees.csv", index=False)
            print("✅ Connexions non optimisées sauvegardées dans 'connexions_non_optimisees.csv'")
        else :
            print("✅ Connexions non optimisées effectuées")
        # Optimisation des connexions en utilisant les critères pondérés
        meilleures_connexions = garder_connexion_optimale(
            objets_connectes,
            df,
            features,
            poids_temps=args.poids_temps,
            poids_distance=args.poids_distance,
            poids_ressemblance=args.poids_ressemblance,
            bonus_cible_source=args.bonus_cible_source
        )
        # Typage propre
        meilleures_connexions = meilleures_connexions.astype({
            "objet_source": int,
            "objet_cible": int,
            "cluster": int
        })

        # Regrouper les connexions en chaînes de trajectoires
        connexions_groupées = reconstituer_trajectoires(meilleures_connexions)

        if args.debug:
            # Export au format souhaité
            connexions_groupées.to_csv("connexions_valides.csv", sep=';', index=False)
            print("✅ Connexions groupées sauvegardées dans 'connexions_valides.csv' 🧩")
        else :
            print("✅ Connexions groupées effectuées")

        # Sauvegarder le fichier des trajectoires reconstituées avec filtrage par durée
        seuil_duree = args.time_min_reconstitute  # Exemple de seuil de durée (en temps)
        save_reconstituted_file(args.csv_path, connexions_groupées, df, seuil_duree)


