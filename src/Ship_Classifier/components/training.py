import time
import os
import urllib.request as request
from zipfile import ZipFile
import torch
import torch.nn as nn
from torchvision import models,transforms,datasets
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Ship_Classifier.entity.config_entity import TrainingConfig
from pathlib import Path



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_base_model(self):
        # Load a pre-trained ResNet-18 model
        self.model = models.resnet18(pretrained=True)
        
        # Modify the final fully connected layer to match the number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, len(self.config.class_names))
        
        # Load the updated base model if it exists
        if self.config.updated_base_model_path.exists():
            self.model.load_state_dict(torch.load(self.config.updated_base_model_path, map_location=self.device))
        
        self.model = self.model.to(self.device)

    def train_valid_loader(self):
        # Define transformations for the training and validation datasets
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.config.params_image_size[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        valid_transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[0]),
            transforms.CenterCrop(self.config.params_image_size[0]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Datasets and data loaders
        train_dataset = datasets.ImageFolder(self.config.training_data, transform=train_transform)
        valid_dataset = datasets.ImageFolder(self.config.training_data, transform=valid_transform)

        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.config.batch_size, shuffle=False)
    
    def display_image_counts(self,path: Path):
    # List of all class folders
     classes = [entry for entry in path.iterdir() if entry.is_dir()]

    # Iterate through each class folder and count the images
     for cls in classes:
        num_images = len(list(cls.glob('*.jpg'))) + len(list(cls.glob('*.png')))  # Update extensions if needed
        print(f"Class: {cls.name}, Number of images: {num_images}")

    def save_model(self, path: Path):
        torch.save(self.model.state_dict(), path)

    def train(self, callback_list: list = None):
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        for epoch in range(self.config.num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(self.train_loader.dataset)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            corrects = 0
            with torch.no_grad():
                for inputs, labels in self.valid_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)

                    pred = torch.argmax(outputs, 1)
                    corrects += torch.sum(pred == labels.data)

            val_loss = val_loss / len(self.valid_loader.dataset)
            val_acc = corrects.double() / len(self.valid_loader.dataset)

            print(f'Epoch {epoch}/{self.config.num_epochs - 1}, '
                  f'Train Loss: {epoch_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            # You can include the callback logic here if you need to apply any callbacks

        self.save_model(self.config.trained_model_path)
        writer = SummaryWriter(log_dir='artifacts/prepare_callbacks/tensorboard_log_dir')