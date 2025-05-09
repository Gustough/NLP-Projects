import pandas as pd
import random
import csv

def bootstrap_csv(input_file, output_file, target_rows=100):
    # Read the existing CSV file
    df = pd.read_csv(input_file)
    
    # Extract header and data
    header = df.columns.tolist()
    data = df.values.tolist()
    
    # Get the number of original rows
    original_rows = len(data)
    if original_rows == 0:
        print("The input CSV file is empty.")
        return
    
    # Determine starting session count and participant index
    last_session = data[-1][0]
    last_session_num = int(last_session.split("-")[1].split("_")[0])
    last_participant = last_session.split("_")[-1]
    participant_order = ["A", "B", "C"]
    last_index = participant_order.index(last_participant)
    
    new_data = []
    session_num = last_session_num
    participant_index = (last_index + 1) % 3
    
    while len(new_data) + original_rows < target_rows:
        row = random.choice(data).copy()
        
        # Update session ID and participant correctly
        if participant_index == 0:
            session_num += 1  # Increment session number every three rows
        participant = participant_order[participant_index]
        row[0] = f"SESS-{session_num:03d}_Physical_{participant}"
        participant_index = (participant_index + 1) % 3
        
        # Introduce minor variations to numerical values and round accordingly
        row[1] = round(max(0, min(1, row[1] * (1 + random.uniform(-0.05, 0.05)))), 3)  # First variable (percentage, 3 decimals)
        row[2] = round(max(0, min(1, row[2] * (1 + random.uniform(-0.05, 0.05)))), 5)  # Second variable (percentage, 5 decimals)
        row[3] = max(0, min(20, round(row[3] + random.uniform(-1, 1))))  # Third variable (0-20 natural number)
        row[4] = max(0, min(20, round(row[4] + random.uniform(-1, 1))))  # Fourth variable (0-20 natural number)
        row[5] = round(max(-1, min(1, row[5] + random.uniform(-0.1, 0.1))), 5)  # Fifth variable (-1 to 1 range, 5 decimals)
        row[6] = round(max(0, min(1, row[6] * (1 + random.uniform(-0.05, 0.05)))), 3)  # Sixth variable (normalized score, 3 decimals)
        row[7] = round(max(0, min(1, row[7] * (1 + random.uniform(-0.05, 0.05)))), 3)  # Seventh variable (normalized score, 3 decimals)
        
        new_data.append(row)
    
    # Create a new DataFrame with the generated data
    new_df = pd.DataFrame([header] + data + new_data)
    
    # Save to output CSV file
    new_df.to_csv(output_file, index=False, header=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Successfully generated {len(new_df) - 1} rows and saved to {output_file}")

# Usage
bootstrap_csv('input.csv', 'output.csv')
