from Ship_Classifier.constants import *
from Ship_Classifier.utils.common  import read_yaml,create_directories
from Ship_Classifier.entity.config_entity import DataIngestionConfig,PrepareBaseModelConfig,PrepareCallbacksConfig,TrainingConfig,EvaluationConfig
import os
from pathlib import Path


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        # Load configurations and parameters from YAML files
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
         # Print the paths to check if they are correct
        print(f"CONFIG_FILE_PATH: {config_filepath}")
        print(f"PARAMS_FILE_PATH: {params_filepath}")
        self.CLASS_NAMES=['Cargo', 'Cruise', 'Carrier', 'Military', 'Tanker']
        create_directories([self.config.artifacts_root])

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
        



    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
            freeze_all=config.get("freeze_all", True),  # Default to True if not specified
            freeze_till=config.get("freeze_till", None)
        )

        return prepare_base_model_config
    
    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([
            Path(model_ckpt_dir),
            Path(config.tensorboard_root_log_dir)
        ])

        prepare_callback_config = PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath),
            epoch=self.params.EPOCHS
        )

        return prepare_callback_config
    
    
    def get_training_config(self) -> TrainingConfig:
       training = self.config.training
       prepare_base_model = self.config.prepare_base_model
       params = self.params
       
       # Update the path to match the directory structure shown in the screenshot
      # training_data = os.path.join(self.config.data_ingestion.unzip_dir, "extracted_data", "Images")
       create_directories([
         Path(training.root_dir)
    ])

       return TrainingConfig(
        root_dir=Path(training.root_dir),
        trained_model_path=Path(training.trained_model_path),  # This now points to the correct image folder structure
        updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
        training_data = Path(os.path.join(self.config.data_ingestion.unzip_dir)),
        batch_size=params.BATCH_SIZE,
        num_epochs=params.epochs,
        learning_rate=params.LEARNING_RATE,
        params_is_augmentation=params.AUGMENTATION,
        params_image_size=params.IMAGE_SIZE,
        class_names=self.config.CLASS_NAMES 
        #base_model_dir=prepare_base_model.model_dir,
        #base_model_name=prepare_base_model.model_name,
        # Call this function in your training code

    )
   
        
        
    def get_validation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
        path_of_model="artifacts/training/model.pth",  # Updated model path to PyTorch format
        training_data="artifacts/data_ingestion/extracted_data",  # Update path to your training images
        all_params=self.params,
        params_image_size=self.params.IMAGE_SIZE,
        params_batch_size=self.params.BATCH_SIZE,
        CLASS_NAMES=['Cargo', 'Cruise', 'Carrier', 'Military', 'Tanker'],  # Add class names
        params_classes=self.params.CLASSES
    )
        return eval_config   
    