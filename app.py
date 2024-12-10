import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn

# Anushka's Part


# Prakhar's Part


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
