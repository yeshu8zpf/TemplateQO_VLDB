import itertools
import subprocess
import yaml
import os

database = 'imdb'
# Set the hyperparameter search space for grid search
hidden_dim_values = [288]
num_layers_values = [6]
lr_values = [1e-4]

# Load the existing config file
config_file = f'configs/{database}.yaml'

def get_subdirectories(directory):
    # Get all entries in the directory (including files and subdirectories)
    entries = os.listdir(directory)
    # Filter out the subdirectories
    subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
    return subdirectories

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

template_train_dir = 'data/imdb/train'
template_paths = get_subdirectories(template_train_dir)

# Iterate through each parameter combination
for i in range(len(template_paths)):
    print(f"Running with templates ({' '.join([template_paths[j] for j in range(i+1)])})")

    # Modify the parameters in the config file
    config['prepare']['used_templates'] = [template_paths[j] for j in range(i+1)]
    config['log']['run'] = i + 1

    # Write the modified config back to the file
    with open(config_file, 'w') as f:
        yaml.dump(config, f)

    # Run the main.py script and pass the config file
    subprocess.run(['python', 'main_rank.py', '--database', 'imdb'])
