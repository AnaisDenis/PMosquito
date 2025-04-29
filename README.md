# PMosquito – Reconstructing Mosquito Trajectories

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


 Installation

First, clone the PMosquito folder using:

	git clone <repository_url>

It is recommended to use a virtual environment:
```
python -m venv venv
source venv/bin/activate  # For Linux/macOS

```

```
venv\Scripts\activate     # For Windows

```

Then, install the required dependencies:

	pip install -r requirements.txt

Make sure the following packages are installed: pandas, numpy, scikit-learn, matplotlib, seaborn.

 Project Structure

The folder is organized as follows:
```
 PMosquito/
├── main.py                    # Main script
├── utils.py                   # Utility functions (clustering, calculations, visualizations)
├── requirements.txt           # Python dependencies
└── jeu_test.csv               # Example dataset (to be provided)
```


 Output Files

The program generates:

    your_filename_with_features.csv: Enriched data including:
	- 'Speed' : overall speed
        - 'AXSplined', 'AYSplined', 'AZSplined' : acceleration components at time t in x, y, and z
	- 'Acceleration': overall acceleration
        - 'TangentialAcceleration' : tangential acceleration
	- 'Curvature': curvature of the trajectory
	- 'DistanceTravelled' : distance traveled between two points

    your_filename_reconstitute.csv: Data with updated trajectory identifiers (when a trajectory is considered a continuation of another)

 Available Parameters

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

 Run Example

Here's an example command to run the program:

	C:\Your_path_to\PMosquito\ > python main.py path_to_your_file.csv --seuil_temps 0.4 --seuil_distance 0.2 --debug --time-min-reconstitute 10.0


In this example:

- Connected mosquitoes must be no more than 0.2m apart and within 0.4s of each other.

- The debug flag enables detailed logging and intermediate results.

- Trajectories shorter than 10 seconds are excluded from the final CSV output.

 Contact

For questions or suggestions, please contact:
olivier.roux@ird.fr

Project developed as part of a Master's thesis on mosquito behavior analysis.
