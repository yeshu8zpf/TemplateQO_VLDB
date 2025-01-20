import itertools
import subprocess
import yaml
import os
import copy
from easydict import EasyDict

# Define the database and config file path
database = 'imdb'
config_file = f'configs/{database}.yaml'

def load_config(file_path):
    """Load YAML configuration from the given file path."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config, file_path):
    """Save YAML configuration to the given file path."""
    with open(file_path, 'w') as f:
        yaml.dump(dict(config), f)

def run_experiment(config, description):
    """Run the main_rank.py script with the current configuration."""
    print(f"Starting experiment: {description}")
    subprocess.run(['python', 'main_rank.py', '--database', database], check=True)
    print(f"Completed experiment: {description}\n")

def main():
    # Load the original configuration
    original_config = load_config(config_file)
    
    # Define all template IDs (0 to 9)
    all_template_ids = list(range(10))  # Templates 0 through 9
    
    for i in all_template_ids:
        # Define current pretrain and new template IDs
        new_template_id = i
        pretrain_template_ids = [tid for tid in all_template_ids if tid != new_template_id]
        new_template_ids = [new_template_id]
        
        # Create a deep copy of the original config to avoid mutations
        config = copy.deepcopy(original_config)
        
        # ----- Pretraining Configuration -----
        config['prepare']['used_templates'] = [f'template{tid}' for tid in pretrain_template_ids]
        config['log']['run'] = f'pretrain_ex{new_template_id}'
        config['finetune']['finetune_path'] = ''
        
        # Save and run pretraining
        save_config(config, config_file)
        run_experiment(config, f"Pretrain with templates {pretrain_template_ids}")
        
        # ----- Finetuning Configuration -----
        config['finetune']['finetune_path'] = f'models/{database}.pth'
        config['finetune']['new_templates'] = new_template_ids
        config['log']['run'] = f'finetune_{new_template_id}'
        config['prepare']['used_templates'] = [f'template{tid}' for tid in new_template_ids]
        
        # Save and run finetuning
        save_config(config, config_file)
        run_experiment(config, f"Finetune with new template {new_template_id}")
        
        # ----- Additional 'Only' Experiment Configuration -----
        config['log']['run'] = f'only{new_template_id}'
        config['prepare']['used_templates'] = [f'template{tid}' for tid in new_template_ids]
        config['finetune']['finetune_path'] = ''
        
        # Save and run the 'only' experiment
        save_config(config, config_file)
        run_experiment(config, f"Only run with new template {new_template_id}")

if __name__ == "__main__":
    main()
