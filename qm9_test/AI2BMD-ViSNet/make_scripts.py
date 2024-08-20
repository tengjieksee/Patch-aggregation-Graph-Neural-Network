


task_list = ["dipole_moment", "isotropic_polarizability", "homo", "lumo", "gap", "electronic_spatial_extent", "zpve", "energy_U0", "energy_U", "enthalpy_H", "free_energy", "heat_capacity"]

i_num = 0
for task in task_list:
    line = f'''CUDA_VISIBLE_DEVICES={i_num} python train.py --conf ./examples/ViSNet-QM9.yml --dataset-arg {task} --dataset-root ./data_folder --log-dir ./log/output_m_2-QM9_task_{i_num}%'''
    print(line)
    i_num += 1
