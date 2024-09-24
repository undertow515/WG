import yaml
import os
from pathlib import Path
class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        self.config = config
        self.config_path = config_path
        
        model_config = config['model']
        for key, value in model_config.items():
            setattr(self, key, value)
        data_config = config['data']
        for key, value in data_config.items():
            setattr(self, key, value)
        train_config = config['trainer']
        for key, value in train_config.items():
            setattr(self, key, value)
        root_config = config['root']
        for key, value in root_config.items():
            setattr(self, key, value)
        self.root_dir = os.path.join(self.runs_dir, self.experiment_name)
        self.checkpoint_dir = f"{self.root_dir}/checkpoints"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
    def save_config_yaml(self, path):
        pass
        