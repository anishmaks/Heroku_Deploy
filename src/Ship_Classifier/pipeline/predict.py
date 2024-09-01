import torch
from torchvision import transforms
from PIL import Image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        #image_names = self.filename
        
        # Define the class names
        self.classes = ['Carrier', 'Cruise', 'Cargo', 'Tanker', 'Military']
        
        # Load the model
        #self.model = torch.load(self.model_path, map_location=self.device)
        #self.model.eval()  # Set model to evaluation mode
    
    def load_model(self):
        
        # load model
        model = self.load_model(os.path.join("artifacts","training", "model.pth"))
        model.eval()
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