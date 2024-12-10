import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn

# Anushka's Part
class CNNModel(nn.Module):
    def _init_(self, num_classes):
        super(CNNModel, self)._init_()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Prakhar's Part
num_classes = 39  # Update to match your dataset
model = CNNModel(num_classes=num_classes)
model.load_state_dict(torch.load("plant_disease_model.pth", map_location=torch.device("cpu")))
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Aman's Part
st.title("Plant Disease Detection")
st.write("Upload an image of a plant leaf, and the model will predict its disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            prediction = model(input_tensor)
            predicted_class = torch.argmax(prediction, dim=1).item()

        st.write(f"Predicted Disease Class: {predicted_class}")
    except Exception as e:
        st.error(f"Error: {e}")
