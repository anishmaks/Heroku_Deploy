import logging
from pathlib import Path
from Ship_Classifier import logger
from Ship_Classifier.config.configuration import ConfigurationManager
from Ship_Classifier.components.prepare_base_model import PrepareBaseModel
from Ship_Classifier.components.prepare_callbacks import PrepareCallback
from Ship_Classifier.components.training import Training



STAGE_NAME= "Training"
class ModelTrainingPipeline:
 def __init__(self):
        pass

 def main(self):


    config = ConfigurationManager()

    #Prepare the base model configuration
    prepare_base_model_config = config.get_prepare_base_model_config()
    prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
    model = prepare_base_model.get_base_model()
    full_model,optimizer,loss_fn,predict,evaluate_model=prepare_base_model.update_base_model()
    
    
    print(f"Model: {full_model}")
    print(f"Optimizer: {optimizer}")
    print(f"Loss: {optimizer}")

    # Prepare the callback configuration
    prepare_callbacks_config = config.get_prepare_callback_config()
    prepare_callbacks = PrepareCallback(
        config=prepare_callbacks_config,
        model=full_model,
        optimizer=optimizer,
        loss=loss_fn
    )
    callback_list = prepare_callbacks.get_tb_ckpt_callbacks()


    # Prepare training configurations
    training_config = config.get_training_config()
    
    # Initialize the training class with the training configuration
    training = Training(config=training_config)
    
    # Load the base model (ResNet-18)
    training.get_base_model()
    
    # Prepare data loaders for training and validation
    training.train_valid_loader()
    
    images_path = Path("artifacts/data_ingestion/extracted_data/Images")
    training.display_image_counts(images_path)
    
    
    # Train the model with the specified callbacks
    training.train(callback_list=callback_list)
    
    
if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
    
