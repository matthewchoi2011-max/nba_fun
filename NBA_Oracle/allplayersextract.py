import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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



