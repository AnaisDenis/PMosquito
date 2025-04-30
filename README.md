# PMosquito – Assembly of interrupted trajectory data by videotracking

This project allows for the analysis, grouping, and reconstruction of mosquito trajectories using spatio-temporal data.
It supports both manual and automatic analysis based on clustering and proximity parameters.

---

## Exigence 

To use this code, you need a CSV file in the following format:

| object 	| time 		| XSplined 	| YSplined 	| ZSplined 	| VXSplined 	| VYSplined 	| VZSpline 	|
|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
| *int*  	| *float*   	| *float* 	| *float* 	| *float*  	| *float* 	| *float* 	| *float* 	|
| Identifiant  	| time (s)  	| x position	| y position 	| z position	| x velocity 	| y velocity	| z velocity 	|

Example (excerpt):

| object 	| time 		| XSplined 	| YSplined 	| ZSplined 	| VXSplined 	| VYSplined 	| VZSpline 	|
|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
| 1	 	| 3.151  	| 0.192		| -0.152	|-0.111		| 0.465 	| -0.050	| 0.403		|
| 1 	 	| 3.171		| 0.201		| -0153		| -0.103	| 0.470		|-0.044		| 0.396	 	|


## Installation of Pmosquito

### Open a terminal
A terminal (or command prompt) is a tool that allows you to interact with your computer by typing text commands. Unlike a graphical interface (where you click buttons), the terminal allows you to execute specific instructions, such as launching a Python script, installing libraries, or navigating through your project folders.

**Windows**: Press Windows + R, type cmd, and then press Enter.
You can also use PowerShell or Visual Studio Code's built-in terminal (View menu > Terminal).

**macOS**: Open the Terminal application via Spotlight (Cmd + Space, then type "Terminal").

**Linux**: Use the shortcut Ctrl + Alt + T or search for "Terminal" in your applications menu.

To install Pmosquito, you need Python 

### Installing Python 
For beginners, follow our [Python installation guide](./INSTALLATION.md) to set up Python and your environment.


### Retrieve this file containing the codes and test 

[[Download PMosquito as a ZIP file](https://github.com/AnaisDenis/PMosquito/archive/refs/heads/main.zip)


### Install some packages
Pmosquito works with some packages, you need to install them for this program to work.
In your terminal you need to move to the Pmosquito-main directory
You must write in your terminal: 
	
	cd path_name

**Note**: To find your path:
Open File Explorer.
Go to your project folder.
Click in the address bar at the top: the path will appear.

**Be careful** if your path contains spaces, use the terminal: windows + r

Then you just need to copy and paste the following line into your terminal to install the packages: 

	pip install -r requirements.txt

Make sure the following packages are installed: pandas, numpy, scikit-learn, matplotlib, seaborn.

## Project Structure

The folder is organized as follows:
```
 PMosquito/
├── main.py                    		# Main script
├── utils.py                  		# Utility functions (clustering, calculations, visualizations)
├── requirements.txt           		# Python dependencies
└── Tests
	├── jeu_test_coupe_0.2.csv	# interruption at the same time
	├── jeu_test_extrait_données_réelles.csv #extracted from trajectory data from laboratory swarms
	├── jeu_test_semi_fictif.csv	#trajectory set from laboratory swarm data
	├── jeu_test_trajectoires_fictifs.csv #set of trajectories from mathematical functions aimed at reproducing a swarm
	├── résultat_semi_fictif
		├── jeu_test_semi_fictif_reconstitue.csv 
		└── visualisation.png # swarm swarm visualization 
	├── résultat_trajectoires_fictifs
		├── jeu_test_moustiques_reconstitue.csv 
		└── visualisation.png # swarm swarm visualization
	└── résultat_extrait_données_réeles
		├── debug
			├── connexions_spatiotemporelles.csv 
			├── connexions_valides.csv 
			├── matrice_spatiotemporelle.csv 
			└── PostProc_Filtered_2022_06_23_18_48_35_Splined_avec_features 
		├── graphiques 
			├── histogram_distance.png 
			├── histogram_time.png 
			├── mirrored_duration_histogram.png
			└── reconstitition_graphique.png 
		└── PostProc_Filtered_2022_06_23_18_48_35_Splined_reconstitue 
		

```


## Output Files

The program generates:

    	your_filename_reconstitute.csv: Data with updated trajectory identifiers (when a trajectory is considered a continuation of another)
    
With option debug :

	connexions_spatiotemporelles.csv 
	connexions_valides.csv # fragments of trajectories that come together
	matrice_spatiotemporelle.csv # result
	your_filename_avec_features # add features  
With option graphiques :	

	histogram_distance.png # distance during the gap
	histogram_time.png # gap time 
	mirrored_duration_histogram.png # comparison of durations after reconstitution
	reconstitition_graphique.png #  visual of the durations of the trajectories and their reconstructions

## Available Parameters

You can customize trajectory reconstruction using the following parameters:

| Argument                  | Description                                          	| Valeurs par defaut      |
|---------------------------|-----------------------------------------------------------|-------------------------|
| `csv_path` *(positional)* | Path to the CSV file                          	   	| Required		  | 
| `--seuil_temps`           | Temporal threshold to connect two objects            	|Optional *(0.5)*         | 
| `--seuil_distance`        | Spatial proximity threshold                          	| Optional *(0.3)*        |
| `--n_clusters`            | Number of clusters to use                      	   	| Optional *(10)*         | 
| `--debug`                 | Displays additional info and intermediate results    	| Optional *(False)*      |
| `--poids-temps`           | Weight of the temporal component                 	   	| Optional *(1.0)*        | 
| `--poids-distance`        | Weight of the spatial component                      	| Optional *(1.0)*        | 
| `--poids-ressemblance`    | Intra-cluster similarity weight                      	| Optional *(1.0)*        | 
| `--bonus-cible-source`    | Bonus if the target is also a source                 	| Optional *(0.3)*        |
| `--time-min-reconstitute` | Minimum duration to keep a trajectory                	| Optional *(0.0)*        | 
| `--graphiques	`	    | Save some statistical graphics about the reconstitution  	| Optional *(0.0)*        |

## Run Example

Here's an example command to run the program:

	C:\Your_path_to\PMosquito\ > python main.py path_to_your_file.csv

To add options, simply enter: "-- 	name of the option	 desired parameter"

	C:\Your_path_to\PMosquito\ > python main.py path_to_your_file.csv --seuil_temps 0.4 --seuil_distance 0.2 --debug --time-min-reconstitute 10.0


In this example:

- Connected mosquitoes must be no more than 0.2m apart and within 0.4s of each other.

- The debug flag enables detailed logging and intermediate results.

- Trajectories shorter than 10 seconds are excluded from the final CSV output.

## Contact

For questions or suggestions, please contact:
olivier.roux@ird.fr

Project developed as part of a Master's thesis on mosquito behavior analysis.
