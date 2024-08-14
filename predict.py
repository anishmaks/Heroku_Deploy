import torch
from model import load_model
import os
from utils import preprocess_image




#model_path = os.path.join('models','resnet18_finetuned_5_classes.pth')
def predict (image_path,model_path=os.path.join('models','resnet18_finetuned_5_classes.pth')):
    
    class_names = ["CARGO", "CARRIER", "TANKER", "MILITARY", "CRUISE"]
    image=preprocess_image(image_path)
    model=load_model(model_path)
    
    with torch.no_grad():
        output=model(image)
        _,predicted=torch.max(output,1)
        
         # Get the corresponding class name
        predicted_class = class_names[predicted.item()]
        
        return predicted_class