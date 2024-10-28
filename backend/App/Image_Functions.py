import torch
import torchvision.transforms as transforms
import cv2
import numpy

def get_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize to match MobileNetV2 input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


# Function to load and preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    transform = get_transform()
    image = transform(image).unsqueeze(0)  # Apply transforms and add batch dimension
    return image


# Function to get image embedding
def generate_embedding(image_path,model):
    image = preprocess_image(image_path)
    with torch.no_grad():  # Disable gradient computation
        embedding = model(image)
    return embedding.flatten().numpy()