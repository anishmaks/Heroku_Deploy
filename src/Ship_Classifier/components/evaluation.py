from urllib.parse import urlparse
from torchvision import transforms, datasets,models
from torch.utils.data import DataLoader
import torch.nn as nn
from Ship_Classifier import config
from Ship_Classifier.utils.common import save_json
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from pathlib import Path

from Ship_Classifier.entity.config_entity import EvaluationConfig

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _valid_loader(self):
        # Define transformations
        transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]),  # Resize images to the desired size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
        ])

        # Load validation dataset
        validation_dataset = datasets.ImageFolder(
            root=self.config.training_data,  # Path to the validation data
            transform=transform
        )
        # Display the number of images in each folder (class)
        class_counts = {class_name: 0 for class_name in validation_dataset.classes}
        for _, label in validation_dataset:
            class_name = validation_dataset.classes[label]
            class_counts[class_name] += 1
        
        for class_name, count in class_counts.items():
            print(f"Found {count} images belonging to class '{class_name}'")

        # Create data loader
        self.valid_loader = DataLoader(
            validation_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False
        )

    @staticmethod
    def load_model(path: str,params_classes:int) -> nn.Module:
        # Load the saved PyTorch model
        model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, params_classes)
       
        state_dict = torch.load(path)
        # Load the state dictionary into the model
        model.load_state_dict( state_dict)
        model.eval()  # Set the model to evaluation mode
        return model

    def evaluation(self):
        # Load the model
        self.model = self.load_model(self.config.path_of_model,self.config.params_classes).to(self.device)

        # Prepare the validation data loader
        self._valid_loader()
        
        # Set up evaluation metrics
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct_predictions = 0
        

        # Evaluate the model
        with torch.no_grad():
            for inputs, labels in self.valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()

        # Calculate average loss and accuracy
        self.average_loss = total_loss / len(self.valid_loader.dataset)
        self.accuracy = correct_predictions / len(self.valid_loader.dataset)

        self.score = {"loss": self.average_loss, "accuracy": self.accuracy}

    def save_score(self):
        
        
        # Save the scores as a JSON file
        save_json(path=Path("scores.json"), data=self.score)