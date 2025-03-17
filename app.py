import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import gdown
from utils import preprocess_image, generate_gradcam, apply_heatmap
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="wide"
)

# Load model with Google Drive integration
@st.cache_resource
def load_detection_model():
    # File paths
    model_path = 'final_model.keras'
    
    # Check if model exists locally first
    if not os.path.exists(model_path):
        # Download model from Google Drive
        with st.spinner("Downloading model file... This may take a minute."):
            # Replace this URL with your shared Google Drive link
            url = 'https://drive.google.com/file/d/1OOefzvwsvstYOXpt325S-py2Vc5KPKM4/view?usp=sharing'  
            gdown.download(url, model_path, quiet=False)
    
    # Load model
    model = load_model(model_path)
    return model

# App title and description
st.title("Skin Cancer Detection")

# Add tabs
tab1, tab2, tab3 = st.tabs(["Detection", "About", "Performance"])

with tab1:
    st.header("Skin Lesion Detection")
    st.write("Upload a dermoscopic image to detect if it's benign or malignant.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.subheader("Original Image")
            st.image(image, width=224)
        
        # Load model
        with st.spinner("Loading model..."):
            model = load_detection_model()
        
        # Preprocess image and make prediction
        with st.spinner("Analyzing image..."):
            img_array = preprocess_image(image)
            prediction = model.predict(img_array)
            
            # Get class prediction
            class_names = ['Benign', 'Malignant']
            predicted_class = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class]) * 100
            
            # Generate Grad-CAM
            try:
                heatmap = generate_gradcam(model, img_array)
                heatmap_img = apply_heatmap(image, heatmap)
                
                with col2:
                    st.subheader("Attention Map")
                    st.image(heatmap_img, width=224)
                    st.caption("Areas the model focused on")
            except Exception as e:
                st.warning(f"Could not generate attention map: {e}")
        
        # Display results
        with col3:
            st.subheader("Prediction")
            
            # Show result with appropriate color
            if predicted_class == 0:  # Benign
                st.success(f"**Result: {class_names[predicted_class]}**")
            else:  # Malignant
                st.error(f"**Result: {class_names[predicted_class]}**")
            
            st.write(f"**Confidence: {confidence:.2f}%**")
            
            # Visualization of probabilities
            st.write("**Probability Distribution:**")
            probs = prediction[0] * 100
            
            # Create probability bars
            fig, ax = plt.subplots(figsize=(5, 2))
            bars = ax.barh(['Benign', 'Malignant'], probs, color=['green', 'red'])
            ax.set_xlim(0, 100)
            ax.set_xlabel('Probability (%)')
            ax.bar_label(bars, fmt='%.1f%%')
            st.pyplot(fig)
            
        st.warning("**Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice.")

with tab2:
    st.header("About")
    st.write("""
    ## Skin Cancer Detection Using ResNet50
    
    This application uses a deep learning model based on ResNet50 architecture to detect whether a skin lesion is benign or malignant from dermoscopic images.
    
    ### How It Works
    
    1. **Image Upload:** User uploads a dermoscopic image of a skin lesion
    2. **Preprocessing:** The image is resized to 224×224 pixels and normalized
    3. **Prediction:** Our trained model analyzes the image and classifies it
    4. **Visualization:** Grad-CAM highlights areas the model focuses on
    
    ### Model Architecture
    
    The model uses a transfer learning approach with ResNet50 as the backbone:
    - Pre-trained ResNet50 base for feature extraction
    - Custom classification layers with dropout and batch normalization
    - Trained on over 76,000 dermoscopic images
    
    ### Important Note
    
    This tool is intended for educational and research purposes only. Always consult a healthcare professional for medical concerns.
    """)

with tab3:
    st.header("Model Performance")
    
    # Performance metrics
    metrics = {
        "Accuracy": "95.59%",
        "Precision": "96.61%",
        "Recall/Sensitivity": "95.01%",
        "Specificity": "96.25%",
        "F1 Score": "95.80%",
        "AUC": "0.9920"
    }
    
    # Display metrics in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Metrics")
        for metric, value in metrics.items():
            st.metric(metric, value)
    
    with col2:
        st.subheader("Confusion Matrix")
        try:
            st.image("assets/confusion_matrix.png")
        except:
            st.write("Confusion matrix image not found")
    
    # ROC Curve
    st.subheader("ROC Curve")
    try:
        st.image("assets/roc_curve.png")
    except:
        st.write("ROC curve image not found")
    
    # Training history
    st.subheader("Training History")
    try:
        st.image("assets/training_history.png")
    except:
        st.write("Training history image not found")

# Footer
st.markdown("---")
st.caption("Created by Dhondhu Kishore - 2023")