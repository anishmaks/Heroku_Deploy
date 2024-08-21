from Ship_Classifier import logger
from Ship_Classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))


STAGE_NAME = "Data Ingestion stage"


if __name__=='__main__':

 try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
 except Exception as e:
        logger.exception(e)
        raise e