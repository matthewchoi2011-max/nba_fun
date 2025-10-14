import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import glob,pandas as pd

processes = []

MAXWORKERS = 8

for i in range(0, 167):
   p = subprocess.Popen(["python", "playerextract.py", str(i)])
   processes.append(p)
   print(f"Started batch {i}")
   if len(processes) >= MAXWORKERS:
      processes[0].wait()
      processes.pop(0)

# wait for remaining
for p in processes:
    p.wait()

   

#merge all error files
all_files = glob.glob("error_list_segment_*.csv")
dfs = [pd.read_csv(f) for f in all_files]
combined = pd.concat(dfs, ignore_index=True)
combined.to_csv("error_list_combined.csv", index=False, encoding="utf-8-sig")


