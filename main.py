import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#import data  
#data.split_dataset()  # Have to run for the first time only with raw dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_dir = "dataset/train"
val_dir = "dataset/validation"
test_dir = "dataset/test"

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),         
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])

train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
validation_dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Ashish's Part


num_classes = len(train_dataset.classes)  
model = CNNModel(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Afiya's Part

# Isha's Part


torch.save(model.state_dict(), "plant_disease_model.pth")
