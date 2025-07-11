import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']

@st.cache_resource
def load_trained_model():
    return load_model("cifar10_classifier.hdf5")

model = load_trained_model()

st.title("🚀 CIFAR-10 Image Classifier")
st.markdown("Upload a 32x32 image and get the model's prediction!")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=200)

    image = image.resize((32, 32))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 32, 32, 3)

    prediction = model.predict(image_array)
    predicted_class = labels[np.argmax(prediction)]

    st.success(f"🧠 Prediction: **{predicted_class}**")
