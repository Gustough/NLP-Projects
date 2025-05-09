# Python files to accompany Speech Technology Project report (Group 19)
## Usage
Running the main.py file will call all the other scripts in sequence
```
python main.py
```
## Included scripts
### create_sim_script.py
This script creates the subjective interaction models from the csv data we were provided with. CSV files of the models are saved in the output directory.
### sim_features_script.py
This script takes subjective interaction models and calculates values for the speech pattern features mentioned in the report
### decision_time.py
This script calculates the decision time feature mentioned in the report. This is seperate from sim_features_script.py because it was a later addition to the project.
### agreement_score.py 
This script calculates the delta scores for each participant as well as assigning each participant a category as mentioned in the report.
### collect_data.py
This script runs the 3 previous scripts and saves all the values as csv files in the output directory
### data_models.py
This script will run the machine learning models and produce the graphs presented in the report.
## Declaration
All python code included in this directory was written by Rhodri Meredith and Gustav Engelmann is part of our Speech Technology project, with the exception of the function: normalised_kendall_tau_distance in script: agreement_score_script.py which was taken from https://en.wikipedia.org/wiki/Kendall_tau_distance