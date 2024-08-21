from Ship_Classifier.constants import *
from Ship_Classifier.utils.common  import read_yaml,create_directories
from Ship_Classifier.entity.config_entity import DataIngestionConfig
import os

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        # Load configurations and parameters from YAML files
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
         # Print the paths to check if they are correct
        print(f"CONFIG_FILE_PATH: {config_filepath}")
        print(f"PARAMS_FILE_PATH: {params_filepath}")

        # Check if the config file exists
        if not os.path.exists(config_filepath):
            raise FileNotFoundError(f"Config file not found at: {config_filepath}")
        if not os.path.exists(params_filepath):
            raise FileNotFoundError(f"Params file not found at: {params_filepath}")
        
         # Load the YAML files
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        # Create the root directory for artifacts if it doesn't exist
        create_directories([self.config['artifacts_root']])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        # Extract the data ingestion configuration from the loaded config
        data_ingestion_config = self.config['data_ingestion']
        
        
        # Check if 'class_dirs' is present
        if 'class_dirs' not in data_ingestion_config:
            raise KeyError("Key 'class_dirs' not found in the configuration")

        # Create the root directory for data ingestion if it doesn't exist
        create_directories([data_ingestion_config['root_dir']])

        # Initialize and return a DataIngestionConfig object with the necessary paths and URLs
        return DataIngestionConfig(
            root_dir=data_ingestion_config['root_dir'],
            source_URL=data_ingestion_config['source_URL'],
            local_data_file=data_ingestion_config['local_data_file'],
            unzip_dir=data_ingestion_config['unzip_dir'],
            class_dirs=data_ingestion_config['class_dirs'] # Renamed to match "extract_dir"
        )