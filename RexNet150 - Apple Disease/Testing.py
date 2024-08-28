import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import timm

image_size = 224
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
transforms = T.Compose([
    T.Resize((image_size, image_size)),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)
])

class CustomModel:
    def __init__(self, model_name, num_classes):
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        
    def load_weights(self, weights_path, device):
        self.model.load_state_dict(torch.load(weights_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def predict(self, image_path, device):
        img = Image.open(image_path).convert("RGB")
        img = transforms(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = self.model(img)
            _, predicted = torch.max(outputs, 1)
            
        return predicted.item()

num_classes = len(class_mapping)
model_name = "rexnet_150"
weights_path = "saved_models/disease_best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomModel(model_name, num_classes)
model.load_weights(weights_path, device)

sample_image_path = "/kaggle/input/apple-leaf/images.jpeg"
predicted_class_index = model.predict(sample_image_path, device)
class_names = list(class_mapping.keys())

print(f"Predicted class: {class_names[predicted_class_index]}")