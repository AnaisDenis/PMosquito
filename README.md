# 🦟 PMosquito – Reconstitution des trajectoires de moustiques

Ce projet permet d'analyser, regrouper et reconstituer des trajectoires de moustiques à partir de données spatio-temporelles. 
Il propose une analyse manuelle ou automatique basée sur des paramètres de clustering et de proximité.

---

## Exigence 

Pour utilisez ce code, vous devez vous munir d'un fichier csv au format suivant :

| object 	| time 		| XSplined 	| YSplined 	| ZSplined 	| VXSplined 	| VYSplined 	| VZSpline 	|
|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
| *int*  	| *float*   	| *float* 	| *float* 	| *float*  	| *float* 	| *float* 	| *float* 	|
| Identifiant  	| instant (s)   | position en x | position en y | position en z | vitesse en x	| vitesse en y	| vitesse en z 	|

exemple (extrait)  : 

| object 	| time 		| XSplined 	| YSplined 	| ZSplined 	| VXSplined 	| VYSplined 	| VZSpline 	|
|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
| 1	 	| 3.151  	| 0.192		| -0.152	|-0.111		| 0.465 	| -0.050	| 0.403		|
| 1 	 	| 3.171		| 0.201		| -0153		| -0.103	| 0.470		|-0.044		| 0.396	 	|


## 🚀 Installation


Après avoir récupérer le dossier PMosquito en le clonant à partir de la commande suivante : 

Il est conseillé de se mettre dans un environnement virtuel :   
```
python -m venv venv
source venv/bin/activate  # Linux/macOS
```

```
venv\Scripts\activate     # Windows
```

Pour faire fonctionner l'outil vous aurez besoin d'installer des dépendances

	pip install -r requirements.txt

Assurez-vous que pandas, numpy, scikit-learn, matplotlib et seaborn sont bien installés.

## 📂 Structure

Ce dossier est structuré de la façon suivante :

```
📁 PMosquito/
├── main.py                    # Script principal
├── utils.py                   # Fonctions utilitaires (clustering, calculs, visualisations)
├── requirements.txt           # Dépendances Python
└── jeu_test.csv               # Exemple de jeu de données (à fournir)
```


### 📦 Sorties générées

    votre_nom_de_fichier_avec_features.csv : Données enrichies par les calculs suivants :
	- 'Speed' : calulcul de la vitesse générale
        - 'AXSplined', 'AYSplined', 'AZSplined' : calcul des accelerations à l'instant t au positions x,y et z respectivement 
	- 'Acceleration': calucul de l'accélération générale
        - 'TangentialAcceleration' : calcul de l'acceleration tangentielle
	- 'Curvature': calcul de la courbure de la trajectoire
	- 'DistanceTravelled' : calcul de la distance effectuée entre deux points

    votre_nom_de_fichier_reconstitute : Données avec un changement d'identifiant pour les trajectoires étant la suite d'une autre 

### ⚙️ Paramètres disponibles

Afin d'ajuster les critères de reconstitution des trajectoires des moustiques, vous pouvez ou devez renseigner les paramètres suivants : 

| Argument                  | Description                                          | Valeurs par defaut      |
|---------------------------|------------------------------------------------------|-------------------------|
| `csv_path` *(positional)* | Chemin vers le fichier CSV                           | Nécessaire              | 
| `--seuil_temps`           | Seuil temporel pour connecter deux objets            | Optionnel *(0.5)*       | 
| `--seuil_distance`        | Seuil spatial de proximité                           | Optionnel *(0.3)*       |
| `--n_clusters`            | Nombre de clusters à utiliser                        | Optionnel *(10)*        | 
| `--debug`                 | Affiche plus d’infos et résultats intermédiaires     | Optionnel *(False)*     |
| `--poids-temps`           | Poids de la composante temporelle                    | Optionnel *(1.0)*       | 
| `--poids-distance`        | Poids de la composante spatiale                      | Optionnel *(1.0)*       | 
| `--poids-ressemblance`    | Poids intra-cluster                                  | Optionnel *(1.0)*       | 
| `--bonus-cible-source`    | Bonus si la cible est également une source           | Optionnel *(0.3)*       |
| `--time-min-reconstitute` | Durée minimale pour garder une trajectoire           | Optionnel *(0.0)*       | 

Pour lancez le programme voici un exemple de commande à inscrire dans son terminal :

	C:\Votre_chemin_d'acces au logiciel\PMosquito\ > python main.py chemin_de_votre_fichier.csv --seuil_temps 0.4 --seuil_distance 0.2 --debug --time-min-reconstitute 10.0

Dans cet exemple : les moustiques assemblez ne peuvent avoir plus de 0.2m d'éloignement avec une différence de temps d'apparition inférieur ou égale à 0.4s. 
Le critère debug sera activé ce qui vous permettra de vérifier les connexions selctionnées et d'autres résultats intermédiaires.
Si un trajet dure moins de 10s il ne sera pas affihcé dans le fichier csv final.  

📬 Contact

Pour toute question ou suggestion : olivier.roux@ird.fr
Projet développé dans le cadre d'un stage de M2 sur l’analyse comportementale des moustiques.
