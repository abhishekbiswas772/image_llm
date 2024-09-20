import torch
import torchvision.transforms as transforms
from torchvision.models import resnet152
from PIL import Image
import numpy as np

class EmbeddingHandler:
    def __init__(self):
        self.model = resnet152(pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def create_embedding(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            image = image.unsqueeze(0)
            with torch.no_grad():
                features = self.model(image)
            return features.squeeze().numpy().astype(np.float32)
        except Exception as e:
            print(f"Failed to create embedding for {image_path}: {e}")
            raise
