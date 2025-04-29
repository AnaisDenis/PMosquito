import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # nécessaire pour les figures 3D
import seaborn as sns

def visualiser_trajectoires(csv_path, cluster_col=None):
    # Chargement
    df = pd.read_csv(csv_path, delimiter=';')
    df.columns = df.columns.str.strip().str.lower()

    # Vérifie que les colonnes de base sont présentes
    for col in ['object', 'xsplined', 'ysplined', 'zsplined']:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante : {col}")

    # Palette de couleurs
    objets = df['object'].unique()
    palette = sns.color_palette("hsv", len(objets))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    if cluster_col and cluster_col.lower() in df.columns:
        clusters = df[cluster_col.lower()].unique()
        palette = sns.color_palette("tab10", len(clusters))
        color_map = {cluster: palette[i] for i, cluster in enumerate(clusters)}

        for cluster in clusters:
            cluster_df = df[df[cluster_col.lower()] == cluster]
            for obj in cluster_df['object'].unique():
                obj_df = cluster_df[cluster_df['object'] == obj]
                ax.plot(obj_df['xsplined'], obj_df['ysplined'], obj_df['zsplined'],
                        label=f"{obj} (C{cluster})", color=color_map[cluster])
    else:
        for i, obj in enumerate(objets):
            obj_df = df[df['object'] == obj]
            ax.plot(obj_df['xsplined'], obj_df['ysplined'], obj_df['zsplined'],
                    label=obj, color=palette[i])

    ax.set_title("Trajectoires 3D des objets")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    #ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.tight_layout()
    plt.show()

# Exemple d’utilisation :
if __name__ == "__main__":
    # Sans cluster :
    visualiser_trajectoires("jeu_test_moustiques.csv")

    # Ou avec cluster :
    # visualiser_trajectoires("resultats_clustering.csv", cluster_col="cluster")
