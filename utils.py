import yaml
import itertools
from typing import Dict, List, Any

def yaml_gen(base_config: Dict[str, Any], param_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate a list of experiment configurations based on a base config and parameter ranges.
    
    :param base_config: Base configuration dictionary
    :param param_ranges: Dictionary of parameters to vary and their possible values
    :return: List of configuration dictionaries
    """
    # Generate all combinations of parameter values
    param_combinations = list(itertools.product(*param_ranges.values()))
    
    # Generate configurations
    configs = []
    for combo in param_combinations:
        # Create a deep copy of the base config
        config = yaml.safe_load(yaml.dump(base_config))
        
        # Update config with new parameter values
        for param, value in zip(param_ranges.keys(), combo):
            keys = param.split('.')
            current = config
            for key in keys[:-1]:
                current = current.setdefault(key, {})
            current[keys[-1]] = value
        
        # Generate experiment name
        exp_name_parts = [f"{key.split('.')[-1]}_{value}" for key, value in zip(param_ranges.keys(), combo)]
        config['root']['experiment_name'] = '_'.join(exp_name_parts)
        
        configs.append(config)
    
    return configs

def save_yaml_configs(configs: List[Dict[str, Any]], output_dir: str):
    """
    Save the generated configurations as YAML files.
    
    :param configs: List of configuration dictionaries
    :param output_dir: Directory to save the YAML files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for i, config in enumerate(configs):
        filename = f"{config['root']['experiment_name']}.yaml"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def write_yaml(data, file_path):
    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file)

def save_yaml(source_path, destination_path):
    data = read_yaml(source_path)
    write_yaml(data, destination_path)

# Example usage
if __name__ == "__main__":
    with open('./configs/test.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    
    param_ranges = {
        'model.hidden_size': [16, 32],
        'trainer.batch_size': [512],
        'model.src_len': [60],
        'data.input_variables': [['Precipitation', 'Intake']],

    }
    
    configs = yaml_gen(base_config, param_ranges)
    save_yaml_configs(configs, 'configs')
    
    print(f"Generated {len(configs)} configurations.")