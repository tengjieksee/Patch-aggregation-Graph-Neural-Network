#srun --jobid=38653275 --pty bash
#module load anaconda/2019.03-Python3.7-gcc5
#source activate /scratch/mo99/tsee0001/miniconda/conda/envs/tensornet_env_1_a40
#screen -ls | awk '/[0-9]+\./ {print $1}' | xargs -I {} screen -S {} -X quit


import subprocess

prev_cmd = "source activate /scratch/mo99/tsee0001/miniconda/conda/envs/tensornet_env_1_a40"#baseline
#prev_cmd = "source activate /scratch/mo99/tsee0001/miniconda/conda/envs/tensornet_env_1"

#task_list = ["dipole_moment", "isotropic_polarizability", "homo", "lumo", "gap", "electronic_spatial_extent", "zpve", "energy_U0", "energy_U", "enthalpy_H", "free_energy", "heat_capacity"]
task_list = ["gap", "electronic_spatial_extent", "zpve", "energy_U0", "enthalpy_H", "free_energy", "heat_capacity"]
gpu_ids = [0,1,2,3,4,5,6]
task_id = [4,5,6,7,9,10,11]
commands = []
i_num = 0
for task in task_list:
    line = f'''{prev_cmd}&CUDA_VISIBLE_DEVICES={gpu_ids[i_num]} torchmd-train --conf ./yaml_folder/m_9-QM9_task_{task_id[i_num]}.yaml --log-dir ./result_folder/output_m_11-QM9_task_{task_id[i_num]}'''
    
    #line = f'''{prev_cmd}&CUDA_VISIBLE_DEVICES={gpu_ids[i_num]} python train.py --conf ./examples/ViSNet-QM9.yml --dataset-arg {task} --dataset-root ./data_folder --log-dir ./log/output_m_10-QM9_task_{i_num}'''
    
    #line = f'''{prev_cmd}&CUDA_VISIBLE_DEVICES={gpu_ids[i_num]} python train.py --conf ./examples/ViSNet-QM9.yml --dataset-arg {task} --dataset-root ./data_folder --log-dir ./log/output_m_2-QM9_task_{i_num} --load-model ./log/output_m_2-QM9_task_{i_num}/result_data/last.ckpt'''
    print(line)
    commands.append(line)
    i_num += 1
    

# Define the commands to be run in separate screen sessions

# Function to create and run a command in a screen session
def create_screen_session(command, session_name):
    subprocess.run(f'screen -dmS {session_name} bash -c "{command}; exec bash"', shell=True, check=True)

# Loop over the commands and create screen sessions
for i, cmd in enumerate(commands):
    if i <20:
        session_name = f"train_task_{i}"
        create_screen_session(cmd, session_name)
        print(i)
    
