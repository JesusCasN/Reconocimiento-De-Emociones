import os
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
from torchvision.models import GoogLeNet_Weights

# Transformaciones para validación
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class GoogleNetWithDropout(nn.Module):
    def __init__(self, num_classes):
        super(GoogleNetWithDropout, self).__init__()
        original_model = models.googlenet(weights=GoogLeNet_Weights.DEFAULT)
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(original_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Mapeo de ID a categoría racial
id2label = {
    0: 'Blanco',
    1: 'Asia Oriental',
    2: 'Latino-Hispano',
    3: 'India, Sudeste Asiático y Medio Oriente',
    4: 'Moreno'
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Inicialización y carga del modelo
num_classes = 5
model_ft = GoogleNetWithDropout(num_classes=num_classes).to(device)
model_ft.load_state_dict(torch.load('archivos/Raza/raza.pth', weights_only=True))
model_ft.eval()

# Función para predecir la clase de una imagen
def predict_image(image_path, model, device, transforms):
    image = Image.open(image_path).convert('RGB')
    image = transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = id2label[predicted.item()]
    return predicted_class

