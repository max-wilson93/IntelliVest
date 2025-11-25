import csv
import os
from datetime import datetime

def log_experiment_result(config, results):
    filename = "monte_carlo_results.csv"
    file_exists = os.path.isfile(filename)
    
    # Combine input config and output results
    row_data = {**config, **results}
    row_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Headers
    headers = ['timestamp', 'final_value', 'return_pct', 'trades'] + list(config.keys())
    
    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
        
    print(f"[Log] Run Saved: {results['return_pct']:.2f}% Return")