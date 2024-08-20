import os

for i in range(12):
    print(i)
    os.system(f"sbatch slurm_launch_task_{i}.script")