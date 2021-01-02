import multiprocessing
import os
import psutil

print(multiprocessing.cpu_count())
print(os.cpu_count())
if os.name == 'posix':
    print(len(os.sched_getaffinity(0)))
elif os.name == 'nt':
    print(len(psutil.Process().cpu_affinity()))
else:
    print("Can't find supported OS")

