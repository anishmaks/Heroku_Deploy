import os
import urllib.request as request
import zipfile
from Ship_Classifier import logger
from Ship_Classifier.utils.common import get_size
import requests
from Ship_Classifier.entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
    def download_file(self):
       if not os.path.exists(self.config.local_data_file):
            try:
                response = requests.get(self.config.source_URL)  # Corrected to 'requests.get'
                response.raise_for_status()  # Raise an HTTPError for bad responses
                with open(self.config.local_data_file, 'wb') as f:
                    f.write(response.content)
                logger.info(f"{self.config.local_data_file} downloaded successfully!")
                
            except requests.exceptions.RequestException as e:  # Corrected to 'requests.exceptions.RequestException'
                logger.error(f"Failed to download the file. Error: {str(e)}")
                raise e
       else:
            file_size = self.get_size(Path(self.config.local_data_file))
            logger.info(f"File already exists with size: {file_size}") 

    def extract_zip_file(self):
        # Ensure the file exists before attempting to extract
        if os.path.exists(self.config.local_data_file):
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(self.config.unzip_dir)  # Changed to 'unzip_dir'
                logger.info(f"Extraction completed successfully into {self.config.unzip_dir}")
            
            # Verifying the folders
            expected_folders = ['Cargo', 'Carrier', 'Cruise', 'Tanker', 'Military']
            for folder in expected_folders:
                folder_path = Path(self.config.unzip_dir) / folder  # Changed to 'unzip_dir'
                if folder_path.exists() and folder_path.is_dir():
                    logger.info(f"Folder '{folder}' exists.")
              #  else:
                #    logger.warning(f"Folder '{folder}' is missing!")
        else:
            logger.error("Zip file does not exist. Please check the download step.")

    def get_size(self, path):
        # Utility method to get the size of a file in bytes
        return os.path.getsize(path)