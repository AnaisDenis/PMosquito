# 🦟 PMosquito – Reconstitution des trajectoires de moustiques

Ce projet permet d'analyser, regrouper et reconstituer des trajectoires de moustiques à partir de données spatio-temporelles. 
Il propose une analyse manuelle ou automatique basée sur des paramètres de clustering et de proximité.

---

## 🚀 Installation


Après avoir récupérer le dossier PMosquito en le clonant à partir de la commande suivante : 

Il est conseillé de se mettre dans un environnement virtuel :   

python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows


Pour faire focntionner l'outil vous aurez besoin d'installer des dépendances

pip install -r requirements.txt

    Assurez-vous que pandas, numpy, scikit-learn, matplotlib et seaborn sont bien installés.

## 📂 Structure

Ce dossier est structuré de la façon suivante :

📁 PMosquito/
├── main.py                    # Script principal
├── utils.py                   # Fonctions utilitaires (clustering, calculs, visualisations)
├── requirements.txt           # Dépendances Python
└── jeu_test.csv               # Exemple de jeu de données (à fournir)


## 📈 Choix du mode de lancement

Il existe deux méthodes de lancement du script : le mode manuel et le mode auto-analyse. Chacunes de ses méthodes présentent ses avantages
et ses inconvénients. Il est donc nécessaire de choisir la méthode la plus adaptée à votre projet. 

### Mode manuel 
Dans le mode manuel vous avez la possibilité de définir de nombreux paramètres permettant de rendre plus ou moins stricte la reconstitution des trajectoires. 
Ce mode est plus rapide que le mode auto-anlyse mais demande à ce que vous sachiez quel valeur attribué à chaques paramètres selon votre projet.

## Mode auto-analyse 
Dans le mode auto-analyse, le programme test différentes valeurs de paramètres et sélectionne la meilleure combinaison pour reconstruire un maximum de trajectoires en gardant 
une fiabilité importante. Ce mode necessite plus de temps de calcul.  
 


### 📦 Sorties générées

    votre_nom_de_fichier_avec_features.csv : Données enrichies par les calculs suivants :
	- 'Speed' : calulcul de la vitesse générale
        - 'AXSplined', 'AYSplined', 'AZSplined' : calcul des accelerations à l'instant t au positions x,y et z respectivement 
	- 'Acceleration': calucul de l'accélération générale
        - 'TangentialAcceleration' : calcul de l'acceleration tangentielle
	- 'Curvature': calcul de la courbure de la trajectoire
	- 'DistanceTravelled' : calcul de la distance effectuée entre deux points

    votre_nom_de_fichier_reconstitute : Données avec un changement d'identifiant pour les trajectoires etant la suite d'une autre 

### ⚙️ Paramètres disponibles

Quelque soit le mode de lancement que vous effectuez vous pouvez ou devez définir les paramètres suivants : 

| Argument                  | Description                                          | Mode Manuel            | Mode Automatique             |
|---------------------------|------------------------------------------------------|-------------------------|-----------------------------|
| `csv_path` *(positional)* | Chemin vers le fichier CSV                           | Nécessaire              | Nécessaire                  |
| `--seuil_temps`           | Seuil temporel pour connecter deux objets            | Nécessaire              | Géré automatiquement        |
| `--seuil_distance`        | Seuil spatial de proximité                           | Nécessaire              | Géré automatiquement        |
| `--n_clusters`            | Nombre de clusters à utiliser                        | Nécessaire              | Géré automatiquement        |
| `--auto-analyse`          | Active le mode auto-analyse                          | Optionnel *(False)*     | Nécessaire *(True)*         |
| `--debug`                 | Affiche plus d’infos et résultats intermédiaires     | Optionnel *(False)*     | Optionnel *(False)*         |
| `--poids-temps`           | Poids de la composante temporelle                    | Optionnel *(1.0)*       | Optionnel *(1.0)*           |
| `--poids-distance`        | Poids de la composante spatiale                      | Optionnel *(1.0)*       | Optionnel *(1.0)*           |
| `--poids-ressemblance`    | Poids intra-cluster                                  | Optionnel *(1.0)*       | Optionnel *(1.0)*           |
| `--bonus-cible-source`    | Bonus si la cible est également une source           | Optionnel *(0.5)*       | Optionnel *(0.5)*           |
| `--time-min-reconstitute` | Durée minimale pour garder une trajectoire           | Optionnel *(0.0)*       | Optionnel *(0.0)*           |




📬 Contact

Pour toute question ou suggestion : olivier.roux@ird.fr
Projet développé dans le cadre d'un stage de M2 sur l’analyse comportementale des moustiques.
