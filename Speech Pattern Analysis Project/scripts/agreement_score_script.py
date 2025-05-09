# Copyright (c) 2025 Rhodri Meredith & Gustav Engelmann
# Licensed under the MIT License. See LICENSE file for details.

import numpy as np
import csv
import os

class get_rankings():
    """Class that takes all the data (as a list) from one meeting in the MEET corpus and returns, as attributes, 
        lists of each ranking relevant to calculating agreement scores.
        Attributes:
         - A_rank_pre: Participant A's pre meeting ranking
         - A_rank_post: Participant A's post meeting ranking
         - B_rank_pre: Participant B's pre meeting ranking
         - B_rank_post: Participant B's post meeting ranking
         - C_rank_pre: Participant C's pre meeting ranking
         - C_rank_post: Participant C's post meeting ranking
         - group_rank: The consensus ranking for the group"""
    
    def __init__(self, data):
        self.meeting_type = data[0][5]
        self.session_num = int(data[0][3][-2:])

        speaker_rankings_pre = {"A": [], "B": [], "C": []}
        speaker_rankings_post = {"A": [], "B": [], "C": [], "group": []}
        for line in data:
            if line[7] == "POST":
                speaker = line[13]
                speaker_rankings_post[speaker].append(line[10])
            elif line[7] == "PRE":
                speaker = line[13]
                speaker_rankings_pre[speaker].append(line[10])
            elif line[7] == "CONSENSUS":
                speaker_rankings_post["group"].append(line[10])
        
        self.A_rank_pre = speaker_rankings_pre["A"]
        self.B_rank_pre = speaker_rankings_pre["B"]
        self.C_rank_pre = speaker_rankings_pre["C"]
        self.A_rank_post = speaker_rankings_post["A"]
        self.B_rank_post = speaker_rankings_post["B"]
        self.C_rank_post = speaker_rankings_post["C"]
        self.group_rank = speaker_rankings_post["group"]

def normalised_kendall_tau_distance(values1, values2):
    """Compute the Normalised Kendall tau distance. Range between 0 and 1. 
        0 is a perfect match, 1 is exactly the opposite order"""
    n = len(values1)
    assert len(values2) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(values1)
    b = np.argsort(values2)
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    return ndisordered / (n * (n - 1))

def get_method3_result(data_segment):
    """Computes the normalised kendell tau distance between the post and consensus ranking
        for each participant. Returns a three item dictionary with the score for each."""
    rankings = get_rankings(data_segment)
    A_score = normalised_kendall_tau_distance(rankings.A_rank_post, rankings.group_rank)
    B_score = normalised_kendall_tau_distance(rankings.B_rank_post, rankings.group_rank)
    C_score = normalised_kendall_tau_distance(rankings.C_rank_post, rankings.group_rank)

    return {"A": round(float(A_score), 3), "B": round(float(B_score), 3), "C": round(float(C_score), 3)}

def get_method4_result(data_segment):
    """Computes the normalised kendell tau distance between each ranking pair
        for each participant. Classifies the participants based on these scores and
        returns a dictionary with each participant and their classification"""
    rankings = get_rankings(data_segment)
    pre_rankings = [rankings.A_rank_pre, rankings.B_rank_pre, rankings.C_rank_pre]
    post_rankings = [rankings.A_rank_post, rankings.B_rank_post, rankings.C_rank_post]
    
    categories = ["IND", "IND", "PERC"]
    result = []
    # Loop over 3 participants
    for i in range(3):
        # Delta 1: Pre and Group. Delta 2: Pre and Post. Delta 3: Group and Post
        delta_1 = normalised_kendall_tau_distance(pre_rankings[i], rankings.group_rank)
        delta_2 = normalised_kendall_tau_distance(pre_rankings[i], post_rankings[i])
        delta_3 = normalised_kendall_tau_distance(post_rankings[i], rankings.group_rank)
        deltas = [delta_1, delta_2, delta_3]
        if (all(x < 0.4 for x in deltas)):
            result.append("DOM")
        else:
            result.append(categories[min(range(len(deltas)), key=deltas.__getitem__)])

    return {"A": result[0], "B": result[1], "C": result[2]}

def get_method1_result(data_segment):
    """Computes the normalised kendell tau distance between the pre and consensus ranking
        for each participant. Returns a three item dictionary with the score for each."""
    rankings = get_rankings(data_segment)
    A_score = normalised_kendall_tau_distance(rankings.A_rank_pre, rankings.group_rank)
    B_score = normalised_kendall_tau_distance(rankings.B_rank_pre, rankings.group_rank)
    C_score = normalised_kendall_tau_distance(rankings.C_rank_pre, rankings.group_rank)

    return {"A": round(float(A_score), 3), "B": round(float(B_score), 3), "C": round(float(C_score), 3)}

def get_method2_result(data_segment):
    """Computes the normalised kendell tau distance between the pre and post meeting ranking
        for each participant. Returns a three item dictionary with the score for each."""
    rankings = get_rankings(data_segment)
    A_score = normalised_kendall_tau_distance(rankings.A_rank_pre, rankings.A_rank_post)
    B_score = normalised_kendall_tau_distance(rankings.B_rank_pre, rankings.B_rank_post)
    C_score = normalised_kendall_tau_distance(rankings.C_rank_pre, rankings.C_rank_post)

    return {"A": round(float(A_score), 3), "B": round(float(B_score), 3), "C": round(float(C_score), 3)}

def main(get_result):
    """ The function reads the csv file containing the rankings from the MEET corpus and returns 
        a the agreement scores that corresponds to the desired method
        Args: 
        - get_result: the function that shall be used to calulcate the agreement score. 
                    get_method1_result, get_method2_result, get_method3_result return 
                    delta 1, 2 or 3 respectively (details in the project report). 
                    get_method4_result returns the classifications detailed in the 
                    classification section of the project report.
        
        Returns:
        - full_result: A dictionary containing a unique meeting ID as the key. The values
                    are another dictionary containing the each participant's agreemnt 
                    score.
        """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file_path = os.path.join(project_root, "data", "MEET - ranking data group 19.csv")
    with open(data_file_path, "r", encoding="utf-8") as f:
        data = list(csv.reader(f))[198:]
        last_idx = 0
        full_results: dict = {}
        while last_idx < len(data):
            data_segment = []
            for i, line in enumerate(data[last_idx:]):
                if line[0]:
                    session_ID = line[2]
                    meeting_type = line[5]
                    data_segment.append(line)
                else:
                    last_idx += i+1
                    break
            else:
                if i + last_idx == len(data)-1:
                    full_results[(session_ID, meeting_type)] = get_result(data_segment)
                    break
        
            full_results[(session_ID, meeting_type)] = get_result(data_segment)
        return full_results

if __name__ =="__main__":
    print(main(get_method1_result))



