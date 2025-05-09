# Copyright (c) 2025 Rhodri Meredith & Gustav Engelmann
# Licensed under the MIT License. See LICENSE file for details.
""""
This script combines the outputs from sim_features_script, agreement_score_script and decision_time
to output a csv file with all these results together.
"""

import scripts.sim_features_script as sim_features_script
import scripts.agreement_score_script as agreement_score_script
import scripts.decision_time as decision_time
import csv
import os

def main(AS_method=3):

    features = sim_features_script.main()
    decision_data = decision_time.main()

    if AS_method == 1:
        scores = agreement_score_script.main(agreement_score_script.get_method1_result)
    elif AS_method == 2:
        scores = agreement_score_script.main(agreement_score_script.get_method2_result)
    elif AS_method == 3:
        scores = agreement_score_script.main(agreement_score_script.get_method3_result)
    elif AS_method == 4:
        scores = agreement_score_script.main(agreement_score_script.get_method4_result)

    # Data order: turn duration, overlap, backchannels by speaker, backchannels to speaker, skewness, decision_score, agreement score
    full_data = {}
    feature_list = ["turn_dur", "overlap", "bc_by_speaker", "bc_to_speaker", "skew", "decision_score", "AS"]
    for ID, speaker_data in features.items():
        for i, speaker in enumerate(["A", "B", "C"]):
            whole_ID = "_".join(ID) + "_" + speaker
            score = scores[ID][speaker]
            decision_score = decision_data[ID][i]
            data = list(speaker_data[speaker]) + [decision_score, score]
            data_dict ={}
            for i, item in enumerate(data):
                data_dict[feature_list[i]] = item
            full_data[whole_ID] = data_dict

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_file_path = os.path.join(project_root, "output", f"model_features_data_method{AS_method}.csv")
    with open(output_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Speaker_ID"] + feature_list) 
        
        for id, results in full_data.items():
            row = [id] + [results.get(feature, "") for feature in feature_list]
            writer.writerow(row)

if __name__ == "__main__":
    main(AS_method=2)
    
