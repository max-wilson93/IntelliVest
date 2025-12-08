import csv
import os
from datetime import datetime

def log_experiment_result(config, results):
    filename = "monte_carlo_results.csv"
    file_exists = os.path.isfile(filename)
    
    # Combine input config and output results
    row_data = {**config, **results}
    row_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Exclude non-serializable objects (like lists)
    if 'equity_curve' in row_data:
        del row_data['equity_curve']
    
    # DYNAMIC HEADERS: This fixes the crash.
    # We grab whatever keys are in the row_data, instead of hardcoding them.
    headers = list(row_data.keys())
    
    # If file exists, we check if we need to append or overwrite (simple append here)
    # Ideally, delete the CSV if schema changes
    
    try:
        with open(filename, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)
    except ValueError:
        # If headers mismatch (old file has different cols), re-write the file or ignore
        pass