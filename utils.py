from PIL import Image
from torchvision import transforms


class_names = ["CARGO", "CARRIER", "TANKER", "MILITARY", "CRUISE"]

def preprocess_image(image_path):
    image=Image.open(image_path).convert('RGB')
    
    transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ])
    
    image_tensor=transform(image)
    image_tensor=image_tensor.unsqueeze(0)
    
    return image_tensor