import sys
import subprocess
import ast
import io
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import pandas as pd
import torch

#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


python_exec = sys.executable

def run_script(run_index):
    try:
        # Run nba_top_75.py and capture output
        output = subprocess.run([python_exec, "nba_top_75_script.py"], capture_output=True,text=True)
        if output.returncode != 0:
            print(f"Run {run_index} failed:")
            print(output.stderr)
        stdout = output.stdout.strip()
        if not stdout:
            # No output, return empty list
            return []
        players = json.loads(stdout)
        return players
    except Exception as e:
        print(f"Error in run {run_index}: {e}")
        return []


# Number of runs
num_runs = 100

# Run in parallel (adjust max_workers based on CPU)
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(run_script, range(1, num_runs + 1)))

# Flatten list of lists
flat_list = [player for sublist in results for player in sublist]

# Count occurrences
counts = Counter(flat_list)

# Sort by count descending
sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

print("Final counts:", sorted_counts)

#save sorted_counts as csv

df = pd.DataFrame(list(sorted_counts.items()), columns=["Player Name", "Count"])
df.to_csv("top_75_1000.csv", sep="\t", index=False)