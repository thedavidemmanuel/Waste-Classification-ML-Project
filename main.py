import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# Set page config
st.set_page_config(page_title="Waste Classification", layout="wide")

# Load the model
model = load_model('model.h5')

# Class names
class_names = ['Organic', 'Recyclable']

# Function to predict the category
def predict_waste_category(model, uploaded_file):
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    return class_names[int(predictions[0] > 0.5)]

# UI Elements
st.title("Waste Classification App")
st.markdown("Identify whether your waste is Organic or Recyclable. Just upload an image!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    prediction = predict_waste_category(model, uploaded_file)
    st.markdown(f"**Prediction**: {prediction}")

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("This app uses a neural network model to classify waste items into Organic or Recyclable categories. It aims to help in waste management and environmental conservation.")
