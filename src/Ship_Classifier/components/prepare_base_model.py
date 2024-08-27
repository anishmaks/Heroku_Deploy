import os
import urllib.request as request
from zipfile import ZipFile
import torch
import torch.nn as nn
from torchvision import models,transforms
import torch.optim as optim
from PIL import Image
from pathlib import Path

from Ship_Classifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
  def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model=None
        self.optimizer = None  # Initialize as needed
        self.loss_fn = None  # Initialize as needed

  def get_base_model(self):
       
        # Load the pretrained ResNet-18 model
       self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
       self.model.fc = nn.Linear(self.model.fc.in_features, self.config.params_classes)
        # Freeze layers if specified
       print(f"Base model loaded with {self.config.params_classes} classes.")
        
        
  @staticmethod      
  def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate,optimizer,loss_fn):
    # Freeze layers as specified
    if freeze_all:
        for param in model.parameters():
            param.requires_grad = False
    elif freeze_till is not None and freeze_till > 0:
        layers = list(model.children())
        num_layers_to_freeze = min(freeze_till, len(layers))  # Ensure we do not exceed the number of layers
        for i, layer in enumerate(layers[:num_layers_to_freeze]):
                for param in layer.parameters():
                    param.requires_grad = False
        
    
    # Modify the final layer to match the number of classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, classes)
    
    # Set up the optimizer
    if optimizer is None:
     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    # Set up the loss function
    if loss_fn is None:
     loss_fn = nn.CrossEntropyLoss()
     
     #Predictions
    def predict(input_tensor):
            with torch.no_grad():  # No need to track gradients for predictions
                model.eval()  # Set the model to evaluation mode
                output = model(input_tensor)
                predictions = torch.argmax(output, dim=1)
            return predictions
    
    print(f"Model prepared with classes={classes}, freeze_all={freeze_all}, freeze_till={freeze_till}, learning_rate={learning_rate}")
    # Summary
    print(model)
    
    #Evaluation
    def evaluate_model(self, dataloader, device):
        self.model.eval()
        correct = 0
        total = 0
        all_pred = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_pred.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy}%')

        # Here you can add more metrics like precision, recall, F1-score
        # For example, you could use sklearn.metrics for more advanced metrics:
        # from sklearn.metrics import classification_report
        # print(classification_report(all_labels, all_pred))

        return accuracy  


    
    return model, optimizer, loss_fn, predict,evaluate_model
         
  def get_model(self):
        return self.model

  def get_optimizer(self):
        return self.optimizer

  def get_loss(self):
        return self.loss_fn
       
    
  @staticmethod
  def save_model(model,path: Path):
        # Save the PyTorch model to the specified path
      if model is not None:
        torch.save(model.state_dict(), path)
        print(f"Model saved at {path}")
      else:
            raise ValueError("Model is not defined. Please check if the model is properly initialized.")
          
        
  def update_base_model(self, freeze_all=True, freeze_till=None):
         self.full_model,self.optimizer,self.loss_fn ,self.predict ,self.evaluate_model =   self._prepare_full_model (
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=freeze_all,
            freeze_till=freeze_till,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            learning_rate=self.config.params_learning_rate)
         
         return self.full_model,self.optimizer,self.loss_fn ,self.predict ,self.evaluate_model
         
         #def get_model(self):
         #  return self.model

         #def get_optimizer(self):
         # return self.optimizer

         #def get_loss(self):
         # return self.loss_fn
        
        # print(f"Model prepared with classes={self._prepare_full_model.classes}, freeze_all={self._prepare_full_model.freeze_all}, freeze_till={self._prepare_full_model.freeze_till}, learning_rate={self._prepare_full_model.learning_rate}")
    # Summary
         print("Full Model :",self.full_model) 
         self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
         
  def load_model(self, path: Path):

      #   Load the model from the specified path.
      
        self.full_model.load_state_dict(torch.load(path), weights_only=True)
        print(f"Model loaded from {path}")