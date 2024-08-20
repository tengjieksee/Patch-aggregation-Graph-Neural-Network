import yaml

# Load the YAML file
with open('template.yaml', 'r') as file: data = yaml.safe_load(file)

# Print the loaded data (for verification)
print("Original data:")
print(data)

# Modify the data as needed
# For example, let's add a new key-value pair or modify an existing one
data['new_key'] = 'new_value'
if 'existing_key' in data:
    data['existing_key'] = 'modified_value'

# Save the modified data to a new YAML file
#with open('output.yaml', 'w') as file:
#    yaml.safe_dump(data, file, default_flow_style=False)

#print("Data has been modified and saved to 'output.yaml'.")

def save_yaml(name, content):
    with open(name, 'w') as file:
        # Write the string to the file
        file.write(content)
    print(name)


task_list = ["dipole_moment", "isotropic_polarizability", "homo", "lumo", "gap", "electronic_spatial_extent", "zpve", "energy_U0", "energy_U", "enthalpy_H", "free_energy", "heat_capacity"]
model_name = "custom_model"
model_id = "m_1"


for i in range(len(task_list)):
    file_path = 'template.yaml'
    with open(file_path, 'r') as file: content = yaml.safe_load(file)
    content["model"] = ""
    content = content.replace('label: dipole_moment', f'label: {task_list[i]}')
    content = content.replace('model: tensornet', f'model: {model_name}')
    save_yaml(name = f"{model_id}-QM9_task_{i}.yaml", content=content)
    

