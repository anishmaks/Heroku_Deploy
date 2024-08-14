import torch
import torch.nn as nn
from torchvision import models
import os

class shipResNetClassifier(nn.Module):
    def __init__(self,num_classes=5):
        super(shipResNetClassifier,self).__init__()
        # define model architecture
        self.model=models.resnet18(pretrained=True)
        self.model.fc=nn.Linear (self.model.fc.in_features,num_classes) #Replacing final layer with the number of classes here
    
    def forward(self,x):
        return self.model(x)
    
def load_model(model_path):
    model_path = os.path.join(os.getcwd(), model_path)
    model =shipResNetClassifier()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    # Load state_dict with strict=False to ignore non-matching keys
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
   # model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')),strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
   
    model.eval()
    return model
    
    