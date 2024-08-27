import logging
from pathlib import Path
from Ship_Classifier import logger
from Ship_Classifier.config.configuration import ConfigurationManager
from Ship_Classifier.components.prepare_base_model import PrepareBaseModel
from Ship_Classifier.components.prepare_callbacks import PrepareCallback
from Ship_Classifier.components.evaluation import Evaluation



STAGE_NAME= "Evaluation Stage"
class EvaluationPipeline:
 def __init__(self):
        pass

 def main(self):
     


 # Initialize the configuration manager
    config = ConfigurationManager()

    # Get the validation configuration
    val_config = config.get_validation_config()

    # Create an Evaluation instance with the validation configuration
    evaluation = Evaluation(val_config)

    # Perform the evaluation
    evaluation.evaluation()

    # Save the evaluation score
    evaluation.save_score()
    
if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e