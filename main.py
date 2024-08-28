
import sys
import os
# Add the src directory to the Python path
#sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))


from Ship_Classifier import logger
from PIL import Image
from Ship_Classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from Ship_Classifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from Ship_Classifier.pipeline.stage_03_training import ModelTrainingPipeline
from Ship_Classifier.pipeline.stage_04_evaluation import EvaluationPipeline


# Add the src directory to the Python path
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
print(sys.path)


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
    
    
STAGE_NAME = "Prepare base model"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   prepare_base_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
      

STAGE_NAME = "Training"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_trainer = ModelTrainingPipeline()
   model_trainer.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Evaluation stage"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evaluation = EvaluationPipeline()
   model_evaluation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
        logger.exception(e)
        raise e