# MNIST Digit Recognizer

## Description
This project is a simple deep learning web application that recognizes handwritten digits (0–9) using a neural network trained on the MNIST dataset. The model is built with **PyTorch** and deployed via **Streamlit** for an interactive web interface. Users can draw digits directly on the web app and get real-time predictions.

## Features
- Trains a neural network on the MNIST dataset with 60,000 images.
- Real-time digit prediction from user input on a drawing canvas.
- Clean and user-friendly web interface using Streamlit.
- Lightweight and efficient model using PyTorch.

## Technologies Used
- Python 3.7+
- PyTorch (for deep learning model)
- torchvision (for MNIST dataset and image transformations)
- Streamlit (for web app interface)
- Pillow (PIL) for image processing

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vaishnavi-kasture/digit-recognizer.git
   cd digit-recognizer

2. Install dependencies:
   
pip install -r requirements.txt


## Usage
To run the Streamlit web app:

streamlit run app.py

- This will open a local webpage where you can draw digits.
- The app predicts the digit in real-time using the trained PyTorch model.

## Project Structure

Digit_Recognition/
    - app.py                # Streamlit web application script
    - train_model.py        # Script to train and save the PyTorch model
    - mnist_model.pth       # Pretrained PyTorch model weights
    - requirements.txt      # Python package dependencies

## Dataset

- Uses the MNIST dataset of handwritten digits.
- Dataset contains 60,000 training images and 10,000 test images.
- Images are 28x28 grayscale.

## Screenshots
![Digit_Classifier](https://github.com/Vaishnavi26-Kasture/Image_Digit_Recognition/blob/main/Digit_Classifier.jpeg?raw=true)
