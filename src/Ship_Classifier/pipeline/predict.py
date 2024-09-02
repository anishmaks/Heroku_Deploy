import torch
import torch.nn as nn
from torchvision import transforms,models
from PIL import Image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        #image_names = self.filename
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define the class names
        self.classes = ['Carrier', 'Cruise', 'Cargo', 'Tanker', 'Military']
        self.model = self.load_model()
        # Load the model
        self.model.to(self.device)  # Move the model to the correct device
        self.model.eval() 
    
    def load_model(self):
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(self.classes))  # Assuming 5 output classes
        # load model
        model.load_state_dict(torch.load(r"artifacts\training\model.pth", map_location=self.device))
        
       # model.eval()
        return model
    
    def predict(self):
        
        
        
        # Define image transformations
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load and preprocess the image
        img = Image.open(self.filename)
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        img_tensor = img_tensor.to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()

        # Interpret the prediction
        prediction = self.classes[predicted_class]
        
        return [{"image": prediction}]