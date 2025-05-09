# Copyright (c) 2025 Rhodri Meredith & Gustav Engelmann
# Licensed under the MIT License. See LICENSE file for details.

import numpy as np
import csv
import math
import os
import pandas as pd

class get_meeting_data():
    """ Class to handle portions of speech pattern data from MEET corpus and create interaction models from them.

    Attributes
    ----------------------
    meeting_data: 
        list of lines from original csv file that are to be included in this portion

    last_meeting_idx: 
        integer giving the index within the csv file where the speech data for this portion ends
        
    length: 
        integer giving the total length of array needed to create interaction model from this portion
    """

    def __init__(self, data, meeting_type, last_meeting_idx):
        meeting_data = []
        for i in range(last_meeting_idx, len(data)):  # Start from last_meeting_idx
            if data[i][1] == meeting_type:
                meeting_data.append(data[i])  # Collect all matching rows
            elif meeting_data:  # If we've started collecting data and reach a different type
                last_meeting_idx = i  # Update last_meeting_idx for next call
                break  # Stop searching
            
        if not meeting_data:
            raise ValueError(f"No data found for meeting type: {meeting_type}")

        self.meeting_data = meeting_data
        self.last_meeting_idx = last_meeting_idx
        self.length = math.ceil((float(meeting_data[-1][3]) * 100) / 5)
    
    def create_IM(self):
        """ Method to create interaction model from the data portion created in the class.
        Args:
            None
        Returns: 
            numpy array of interaction model. Contains rows of each speaker
                 and columns for every time interval of 50ms
        """

        speakers = ["A", "B", "C"]
        oim_array = np.zeros((3, self.length))
        lines_in_intervals = []  # Store all timestamps and corresponding indices

        for time_stamp in np.arange(self.length):
            lines_in_interval = []  # Reset for each timestamp
            time_stamp = float(time_stamp)  # Convert timestamp to float once

            for i, line in enumerate(self.meeting_data):
                turn_start = (float(line[2]) * 100) / 5
                turn_end = (float(line[3]) * 100) / 5

                if turn_start <= time_stamp <= turn_end:
                    lines_in_interval.append(i)

            lines_in_intervals.append((time_stamp, lines_in_interval))

        for item in lines_in_intervals:
            lines_in_data = item[1]
            for line in lines_in_data:
                if self.meeting_data[line][-2] == "SIL":
                    break
                else:
                    oim_array[speakers.index(self.meeting_data[line][-1])][int(item[0])] = 1
        return oim_array

def create_SIM(oim_array):
    speakers = ["A", "B", "C"]
    sim_array = np.zeros_like(oim_array)
    for i in range(len(speakers)):
        oim_array[i] = np.multiply(oim_array[i], 4)
        array_sum = oim_array.sum(axis=0)
        sim_array_row = np.zeros_like(array_sum)
        for j, num in enumerate(array_sum):
            if num == 1 or num == 2:
                sim_array_row[j] = 1
            elif num == 4:
                sim_array_row[j] = 2
            elif num > 4:
                sim_array_row[j] = 3
            else:
                pass
        sim_array[i] = sim_array_row
        oim_array[i] = np.multiply(oim_array[i], 0.25)
    
    return sim_array

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file_path = os.path.join(project_root, "data", "MEET data-group P19.csv")

    with open(data_file_path, "r", encoding="utf-8") as f:
        data = list(csv.reader(f))
        data = data[1:]

    meeting_types = ["digital", "hybrid", "physical"]
    last_meeting_idx = 0
    for i in range(10):
        for item in meeting_types:

            meeting = get_meeting_data(data, item, last_meeting_idx)
            
            IM_array = meeting.create_IM()

            SIM_array = create_SIM(IM_array)

            last_meeting_idx = meeting.last_meeting_idx

            output_file_path = os.path.join(project_root, "output", "SIM_files", f"ss{i}_{item}.csv")
            df = pd.DataFrame(SIM_array)
            df.to_csv(output_file_path, index=False, header=False)

if __name__ == "__main__":
    main()
    
