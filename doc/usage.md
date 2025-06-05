# How use Pmosquito? 

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