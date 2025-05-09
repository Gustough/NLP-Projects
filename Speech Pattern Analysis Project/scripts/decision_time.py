# Copyright (c) 2025 Rhodri Meredith & Gustav Engelmann
# Licensed under the MIT License. See LICENSE file for details.

import os
import csv
import numpy as np
from collections import defaultdict
import pympi.Elan as elan
import pandas as pd
import ast

def extract_decision_time(input_directory, filename):
    results = []
    file_path = os.path.join(input_directory, filename)
    eaf = elan.Eaf(file_path)
    decision_time = 0
    for entry in eaf.get_annotation_data_for_tier("Focus"):
        if entry[2] == "decision":
            decision_time = int(round(((entry[:2][0] + entry[:2][1]) / 2) / 10 / 5))
            results.append(decision_time)
    return results

def get_decision_maker(input_directory_times, input_directory_decisions):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results = defaultdict(list)
    results_post = defaultdict(list)
    speakers = ["A", "B", "C"]
    counter = 0
    for filename in os.listdir(input_directory_times):
        percentages = defaultdict(int)
        file_path_times = os.path.join(input_directory_times, filename)
        with open(file_path_times, "r", encoding="utf-8") as f:
            data = csv.reader(f)
            data = [[float(value) for value in row] for row in data]
            bc = pd.read_excel(os.path.join(project_root, "data", "back_times.xlsx"))
            bc = bc.to_numpy().tolist()  # Convert DataFrame to a nested list of floats
            timestamps = extract_decision_time(input_directory_decisions, filename[:-4] + ".eaf")
            for t in timestamps:
                for i, s in enumerate(speakers):
                    line = np.array(data[i])
                    if line[t] == 2:
                        results[filename].append(s)
                        for b in bc[counter]:
                            if isinstance(b, str):
                                lis = ast.literal_eval(b)
                                lis = [int(x) for x in lis]
                                for item in lis:
                                    if (t - 100) < item < (t + 100) and b != 0:
                                        results[filename].append(speakers[i])
                            elif (t - 100) < b < (t + 100) and b != 0:
                                results[filename].append(speakers[i])
        number_of_decisions = len(results[filename])
        for r in results[filename]:
            percentages[r] += 1
        counter += 1
            
        label = ("SESS-0"+ f"{int(filename[2])+1:02}", filename.split("_")[1][:-4].capitalize())
        results_post[label] = [round((percentages["A"] / number_of_decisions), 3), round((percentages["B"] / number_of_decisions),3), round((percentages["C"] / number_of_decisions), 3)]
    return results_post
                         
        

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_folder_1 = os.path.join(project_root, "output", "SIM_files")
    input_folder_2 = os.path.join(project_root, "data", "elan")
    dt = get_decision_maker(input_folder_1, input_folder_2)
    return dt

if __name__ == "__main__":
    print(main())
