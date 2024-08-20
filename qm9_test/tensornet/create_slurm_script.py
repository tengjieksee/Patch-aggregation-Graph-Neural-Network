import os

# Define the template for the SLURM script
script_template = """#!/bin/bash
#SBATCH --job-name=qm9_tensornet_m_17_task_{task_number}
#SBATCH --account=mo99
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=m3g
module load miniforge3
mamba activate tensornet_env_1
nvidia-smi
CUDA_VISIBLE_DEVICES=0 torchmd-train --conf ./yaml_folder/m_9-QM9_task_{task_number}.yaml --log-dir ./result_folder/output_m_17-QM9_task_{task_number}
"""

# Directory to save the script files
script_directory = './'
os.makedirs(script_directory, exist_ok=True)

# Generate script files for tasks 0 to 11
for task_number in range(12):
    script_content = script_template.format(task_number=task_number)
    script_filename = os.path.join(script_directory, f'slurm_launch_task_{task_number}.script')
    with open(script_filename, 'w') as script_file:
        script_file.write(script_content)

print("Script files have been created.")
