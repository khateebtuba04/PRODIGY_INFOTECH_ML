import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from src.calorie_utils import get_calorie_info

# Page Config
st.set_page_config(
    page_title="NutriScan AI",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2b42 100%);
        color: #ffffff;
    }

    /* Titles */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
        text-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    }

    /* Card Style - Glassmorphism */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.3);
    }

    /* Upload Area */
    .stFileUploader > div > div {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        border: 2px dashed rgba(255, 255, 255, 0.2);
        padding: 20px;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #FF4B4B;
    }

    /* Metrics */
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF9068);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF9068 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }

</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("This AI model identifies 10 different food categories and provides calorie estimates.")
    st.info("Supported classes:\n" + ", ".join([
        "Apple Pie", "Baby Back Ribs", "Baklava", "Beef Carpaccio", 
        "Beef Tartare", "Beet Salad", "Beignets", "Bibimbap", 
        "Bread Pudding", "Breakfast Burrito"
    ]))

# Main Content
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('# 🥗 NutriScan AI')
    st.markdown("#### Intelligent Food Analysis")
    st.write("Upload a photo of your meal to get instant calorie breakdowns and classification.")

# Load Resources
@st.cache_resource
def load_model():
    if os.path.exists('model.h5'):
        return tf.keras.models.load_model('model.h5')
    return None

@st.cache_data
def load_class_indices():
    if os.path.exists('class_indices.json'):
        with open('class_indices.json', 'r') as f:
            return json.load(f)
    return None

model = load_model()
class_indices = load_class_indices()

# Main Interaction Area
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📸 Snap or Upload a Food Image", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Processing Layout
    c1, c2 = st.columns(2)
    
    with c1:
        image = Image.open(uploaded_file)
        # Fix: Replaced use_column_width with use_container_width
        st.image(image, caption='Your Meal', use_container_width=True) 
    
    with c2:
        if not model or not class_indices:
            st.error("⚠️ Model not found! Please ensure training is complete.")
        else:
            with st.spinner('Thinking... 🧠'):
                # Invert class_indices
                class_names = {v: k for k, v in class_indices.items()}
                
                # Preprocess
                img = image.resize((224, 224))
                img_array = np.array(img)
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                predictions = model.predict(img_array)
                predicted_class_idx = np.argmax(predictions)
                confidence = np.max(predictions)
                
                predicted_class = class_names[predicted_class_idx]
                
                # Get Calorie Info
                calorie_info = get_calorie_info(predicted_class)
                
                # Display Results
                st.markdown(f"""
                <div class="glass-card">
                    <h3>Analysis Result</h3>
                    <p style="font-size: 1.2rem; color: #ccc;">Dish Name</p>
                    <div class="metric-value">{predicted_class.replace('_', ' ').title()}</div>
                    <br>
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <p style="color: #ccc;">Calories</p>
                            <h4>{calorie_info['calories']} kcal</h4>
                        </div>
                        <div>
                            <p style="color: #ccc;">Confidence</p>
                            <h4>{confidence*100:.1f}%</h4>
                        </div>
                    </div>
                    <hr style="border-color: rgba(255,255,255,0.1);">
                    <p><i>Serving Size: {calorie_info['unit']}</i></p>
                </div>
                """, unsafe_allow_html=True)
