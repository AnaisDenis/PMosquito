# PMosquito – Assembly of interrupted trajectory data by videotracking

In tracking experiments, brief signal losses can occur. When tracking multiple, closely spaced objects, these interruptions can cause short, fragmented trajectories that are difficult—or even impossible—to assign to the correct object.

PMosquito solves this challenge by reliably reconnecting broken trajectories, delivering longer and more complete paths for tracked object.
Originally developed for 3D video tracking of mosquito swarms within the ANOFEEL project – How ANOpheles Females sEEk maLes? (ANR-15-CE35-0001-01) – PMosquito can process any dataset that records the location of multiple objects in space and time, whether you are studying animal behavior, particle motion, swarm robotics, and more.

A range of parameters can be customized to match the specific requirements of your study model.


<img src="doc/img/IRD.png" width="300" height="100" /> <img src="doc/img/MIVEGEC.png" width="150" height="100" />

---

👉 **Access the documentation site here:** [PMosquito Web Page](https://anaisdenis.github.io/PMosquito/)

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

In your terminal you need to move to the Pmosquito-main directory
You must write in your terminal: 
	
 	cd path_name

**Note**: To find your path:
Open File Explorer.
Go to your project folder.
Click in the address bar at the top: the path will appear.

Then, type :

	python -m venv env

Here, env is the name of the virtual environment. You can choose a different name if you wish.
This creates an env/ folder containing an isolated Python installation.

Then ypu need to activate your virtual environment. Type always in your terminal :

**For Windows**

	.\env\Scripts\activate

**For Linux/macOS**

	source env/bin/activate

Normally you should see your virtual environment name displayed in parentheses at the beginning of the new command line

### Retrieve this file containing the codes and test 

[Download PMosquito as a ZIP file](https://github.com/AnaisDenis/PMosquito/archive/refs/heads/main.zip)

You can extract this folder to the directory of your choice: you will find the .zip file in your downloads. Move it by right-clicking, cutting, and pasting it into the directory of your choice. Then right-click and extract here.

### Install Pmosquito

Once your virtual environment is activated and you are in the project directory (the folder that contains `pyproject.toml`), install Pmosquito and its dependencies with:

    pip install .

This command will install:
- the `Pmosquito` tool,
- all required dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.


## Project Structure

The folder is organized as follows:
```
 PMosquito/
├── main.py                    		# Main script
├── utils.py                  		# Utility functions (clustering, calculations, visualizations)
├── requirements.txt           		# Python dependencies
└── Tests
	├── test_set_fictive.csv	
	├── test_set_semi_fictionnal.csv
	├── test_set_representing_a_tracking_bug.csv
	├── test_set_extract_real_data.csv
	├── result_test_set_fictive
		├── test_set_fictive_reconstitue.csv 
		└── visualisation.png # swarm swarm visualization
	├── result_test_set_semi_fictionnal
		├── test_set_semi_fictionnal_reconstitue.csv 
		└── visualisation.png # swarm swarm visualization 
	└── result_test_set_extract_real_data
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
## Tests files

The test folder includes several different data files.
- test_set_fictive.csv : This file contains 50 initial trajectories divided into 4 fragments. The times associated with the trajectories are between t= 0s and t=30s. Each trajectory is derived from functions allowing the creation of trajectories that can approximate those of a swarm of mosquitoes.
  
- test_set_semi_fictionnal.csv : This file contains 66 initial trajectories divided into 4 fragments. The times associated with the trajectories are between t= 0s and t=30s. Each trajectory is taken from real data but has been repositioned in time and selected because it lasted at least 30sec.

- test_set_representing_a_tracking_bug.csv : These are the same trajectories as the previous file, but the fragments are segmented 0.2 seconds to the 20th second for all trajectories. This file can be compared to a technical tracking bug.

- test_set_extract_real_data.csv : This dataset is an extract from a csv file that represents our real data from mosquito video tracking.



## Output Files

The program generates:

	your_filename_reconstitute.csv # Data with updated trajectory identifiers (when a trajectory is considered a continuation of another)
    
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

	pmosquito path_to_your_file.csv


Make sure you're always in the Pmosquito file: your terminal command line should always start with C:\Your_path_to\PMosquito\ >
If this isn't the case, don't forget to 
	
 	cd path_name

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
