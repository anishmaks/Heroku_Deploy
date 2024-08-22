from Ship_Classifier.config.configuration import ConfigurationManager
from Ship_Classifier.components.prepare_base_model import PrepareBaseModel
from Ship_Classifier import logger
import logging

STAGE_NAME= "Prepare Base Model"

class PrepareBaseModelTrainingPipeline:
  def __init__(self) -> None:
        pass
    
  def main(self):
    logger = logging.getLogger(__name__)
  
    # Initialize the configuration manager
    logger.info("Initializing ConfigurationManager")
    config = ConfigurationManager()
    prepare_base_model_config = config.get_prepare_base_model_config()
    prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
    prepare_base_model.get_base_model()
    prepare_base_model.update_base_model()
    prepare_base_model.save_model(model=prepare_base_model.model,path=prepare_base_model_config.base_model_path)
    prepare_base_model.save_model(model=prepare_base_model.full_model,path=prepare_base_model_config.updated_base_model_path)
    
    
    if __name__ == '__main__':
     try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
     except Exception as e:
        logger.exception(e)
        raise e
