
def save_yaml(name, content):
    with open(name, 'w') as file:
        # Write the string to the file
        file.write(content)
    print(name)


task_list = ["dipole_moment", "isotropic_polarizability", "homo", "lumo", "gap", "electronic_spatial_extent", "zpve", "energy_U0", "energy_U", "enthalpy_H", "free_energy", "heat_capacity"]
model_name = "custom_model"
model_id = "m_9"


for i in range(len(task_list)):
    file_path = 'template_small.yaml'
    with open(file_path, 'r') as file: content = file.read()
    
    content = content.replace('label: dipole_moment', f'label: {task_list[i]}')
    content = content.replace('model: tensornet', f'model: {model_name}')
    save_yaml(name = f"{model_id}-QM9_task_{i}.yaml", content=content)
    

