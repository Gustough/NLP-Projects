# Copyright (c) 2025 Rhodri Meredith & Gustav Engelmann
# Licensed under the MIT License. See LICENSE file for details.

import scripts.create_sim_script
import scripts.collect_data
import scripts.data_models

def main():
    # print("Creating subjective interaction models from MEET corpus data inside /output/SIM_files/ ...")
    # scripts.create_sim_script.main()
    # print("Done!")
    # print("Creating speech pattern features from interaction models with four Agreement score methods inside /output/ ...")
    for i in range(4):
        scripts.collect_data.main(AS_method=i+1)
        pass
    print("Done!")
    while True:
        s = input("Press 1 for linear model, press 2 for random forest classification, press 3 for k-means clustering model, press 4 to exit: ")
        if int(s) == 1:
            print("Creating linear model ...")
            scripts.data_models.linear_regression()
        elif int(s) == 2:
            print("Creating random forest classification model ... ")
            scripts.data_models.run_random_forest()
        elif int(s) == 3:
            print("Running k-means clustering model ...")
            scripts.data_models.unsupervised_learning(check_optimal_k=True, get_boxplot=True, get_anova=True, check_norm=True)
        elif int(s) == 4:
            break
        else:
            print("Not a valid input")
    print("Completed!")
main()