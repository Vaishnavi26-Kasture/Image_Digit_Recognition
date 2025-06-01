import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# Define the same model architecture
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Load model
@st.cache(allow_output_mutation=True)
def load_model():
    model = DigitClassifier()
    model.load_state_dict(torch.load("mnist_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# UI
st.title("🧠 MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (28x28, black background)")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Digit", width=150)

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform(image).unsqueeze(0)

    # Predict
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs.data, 1)
    st.success(f"✅ Predicted Digit: {predicted.item()}")

