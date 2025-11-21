# app.py - Complete Streamlit App Code
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import os

# Hardcode class names from your Colab output (replace with actual 15 vegetable names, sorted alphabetically)
class_names = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']  # Example; update this list!

# Load and build model (recreate architecture + load weights)
@st.cache_resource
def load_model():
    # Recreate VGG16 base (exact as training)
    conv_base = keras.applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )
    
    # Fine-tuning: Freeze up to block5_conv1, unfreeze after (exact as training)
    conv_base.trainable = True
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    
    # Build Sequential model (exact as training)
    model = keras.Sequential([
        conv_base,
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(15, activation='softmax')
    ])
    
    # Compile (use same as training; adjust if needed)
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Load the saved weights

    weights_path = hf_hub_download(
    repo_id="Peacfl/Vegetable-image-classification",
    filename="vegetable_classifier_weights.weights.h5"
)

model = tf.keras.models.load_model(model_path)
    model.load_weights(weights_path)
    
    return model

model = load_model()

# Preprocess function (adapted from Colab)
def preprocess_image(image):
    """
    Preprocess uploaded image for prediction.
    - Resize to 150x150
    - Normalize to [0,1]
    - Add batch dimension
    """
    img = image.convert('RGB').resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit App UI
st.title("Vegetable Image Classifier")
st.write("Upload an image of a vegetable, and it will predict which one it is using a fine-tuned VGG16 model")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")
    
    # Preprocess and predict
    input_image = preprocess_image(image)
    predictions = model.predict(input_image)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]
    
    # Display results
    st.success(f"**Predicted Vegetable:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2%}")
    
    # Optional: Show top 3 predictions
    top_3_idx = np.argsort(predictions[0])[::-1][:3]
    st.subheader("Top 3 Predictions:")
    for i, idx in enumerate(top_3_idx):
        class_name = class_names[idx]
        conf = predictions[0][idx]
        st.write(f"{i+1}. {class_name}: {conf:.2%}")
    
    # Optional: Show all class names for reference
    with st.expander("View All Vegetable Classes"):
        st.write(class_names)

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.write("Trained on 15 vegetable classes from the Vegetable Image Dataset.")
    st.write("Model: Fine-tuned VGG16 (ImageNet weights).")
    st.write("Input size: 150x150 pixels.")
    if st.button("Reload Model"):
        st.cache_resource.clear()

        st.rerun()
