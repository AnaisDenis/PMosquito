# PMosquito – Assembly of interrupted trajectory data by videotracking

This project allows for the analysis, grouping, and reconstruction of mosquito trajectories using spatio-temporal data.
It supports both manual and automatic analysis based on clustering and proximity parameters.

---

## Exigence 

To use this code, you need a CSV file in the following format:

- **object**: an integer identifier representing an object tracked over time.

- **time**: a decimal number representing the elapsed time in seconds.

- **XSplined**: the object's position along the X-axis (horizontal coordinate), expressed as a decimal number.

- **YSplined**: the object's position along the Y-axis (vertical coordinate), expressed as a decimal number.

- **ZSplined**: the object's position along the Z-axis (depth or height), expressed as a decimal number.

- **VXSplined**: the object's velocity along the X-axis, expressed as a decimal number.

- **VYSplined**: the object's velocity along the Y-axis, expressed as a decimal number.

- **VZSplined**: the object's velocity along the Z-axis, expressed as a decimal number.

Each row of the file therefore corresponds to the state of an object at a given moment, including its spatial position and velocity in all three dimensions.

Example (excerpt):

| object 	| time 		| XSplined 	| YSplined 	| ZSplined 	| VXSplined 	| VYSplined 	| VZSplined 	|
|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
| 1	 	| 3.151  	| 0.192		| -0.152	|-0.111		| 0.465 	| -0.050	| 0.403		|
| 1 	 	| 3.171		| 0.201		| -0153		| -0.103	| 0.470		|-0.044		| 0.396	 	|


## Installation of Pmosquito

### Open a terminal
A terminal (or command prompt) is a tool that allows you to interact with your computer by typing text commands. Unlike a graphical interface (where you click buttons), the terminal allows you to execute specific instructions, such as launching a Python script, installing libraries, or navigating through your project folders.

**Windows**: Press Windows + R, type cmd, and then press Enter.

**macOS**: Open the Terminal application via Spotlight (Cmd + Space, then type "Terminal").

**Linux**: Use the shortcut Ctrl + Alt + T or search for "Terminal" in your applications menu.

To install Pmosquito, you need Python 

### Check if Python is Installed

Type the following command:

	python --version

or 

	python3 --version

If you see something like Python 3.x.x, Python is already installed. Otherwise, proceed to the next step.

### Install Python

**For Windows**

Go to https://www.python.org/downloads/windows/

Click Download Python 3.x.x

IMPORTANT: On the first installation screen, check the box "Add Python to PATH"

Click Install Now

After installation, reopen your terminal and check:

	python --version

**For macOS**

If needed, install Homebrew

Use the following command:

	brew install python


**For Linux (Debian, Ubuntu, etc.)**

Use the following commands:

	sudo apt update
	sudo apt install python3 python3-venv python3-pip

### Install pip (if not already installed)

Most modern Python installations include pip. To check:

	pip --version

or

	pip3 --version

If it's missing:

On Windows, reinstall Python and ensure "Install pip" is checked

On Linux, use:

	sudo apt install python3-pip

### Virtual Environment

It is recommended to use a virtual environment:

- Each virtual environment contains its own version of Python and its own packages, independent of other projects or the system installation. This prevents a package update for one project from breaking another.
- You can install exactly the versions of libraries needed for one project without affecting others. Perfect for replicating an environment on another machine.


To do this, in your terminal, go to your project folder, then type:

	python -m venv env

Here, env is the name of the virtual environment. You can choose a different name if you wish.
This creates an env/ folder containing an isolated Python installation.

**For Windows**

	.\env\Scripts\activate

**For Linux/macOS**

	source env/bin/activate


### Retrieve this file containing the codes and test 

[Download PMosquito as a ZIP file](https://github.com/AnaisDenis/PMosquito/archive/refs/heads/main.zip)

You can extract this folder to the directory of your choice: you will find the .zip file in your downloads. Move it by right-clicking, cutting, and pasting it into the directory of your choice. Then right-click and extract here.

### Install some packages
Pmosquito works with some packages, you need to install them for this program to work.
In your terminal you need to move to the Pmosquito-main directory
You must write in your terminal: 
	
	cd path_name

**Note**: To find your path:
Open File Explorer.
Go to your project folder.
Click in the address bar at the top: the path will appear.


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
