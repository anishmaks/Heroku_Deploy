from Ship_Classifier.config.configuration import ConfigurationManager
from Ship_Classifier.components.data_ingestion import DataIngestion
from Ship_Classifier import logger
import logging

STAGE_NAME= "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
  def __init__(self) -> None:
        pass
    
  def main(self):
    logger = logging.getLogger(__name__)
  
    # Initialize the configuration manager
    logger.info("Initializing ConfigurationManager")
    config = ConfigurationManager()
    
    # Retrieve the data ingestion configuration
    # Retrieve the data ingestion configuration
    logger.info("Retrieving data ingestion configuration")
    data_ingestion_config = config.get_data_ingestion_config()
    
    # Initialize the DataIngestion process with the retrieved configuration
    logger.info("Initializing DataIngestion")
    data_ingestion = DataIngestion(config=data_ingestion_config)
    
    # Download the file if it doesn't already exist locally
    logger.info("Downloading file")
    data_ingestion.download_file()
    
    # Extract the downloaded zip file to the specified directory
    logger.info("Extracting zip file")
    data_ingestion.extract_zip_file()

if __name__=='__main__':
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj=DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx=================x")
    except Exception as e:
        logger.exception(e)
        raise e
    