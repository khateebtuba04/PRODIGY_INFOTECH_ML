import streamlit as st
import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image
import time

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CUSTOM CSS FOR MODERN UI
# =============================================================================
st.markdown("""
<style>

/* Modern Dark Theme with Glassmorphism */
.stApp {
    background-color: #0E1117;
    color: #FAFAFA;
}

/* Main content area */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Glassmorphism Cards */
.metric-card, .confidence-container, .header-container {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border-radius: 16px;
    padding: 1.5rem;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.metric-card:hover, .confidence-container:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0, 242, 254, 0.1); /* Neon blue glow */
    border-color: rgba(0, 242, 254, 0.3);
}

/* Header Specifics */
.header-container {
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
}

.main-title {
    font-size: 3rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #00f2fe, #4facfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #cbd5e1;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

/* Metrics */
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #00f2fe; /* Neon Blue */
}

.metric-label {
    color: #cbd5e1;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Prediction cards */
.prediction-cat, .prediction-dog {
    padding: 2rem;
    border-radius: 16px;
    font-size: 2rem;
    font-weight: 800;
    text-align: center;
    color: #0E1117; /* Dark text for contrast on bright bg */
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    border: 2px solid rgba(255,255,255,0.2);
}

.prediction-cat {
    background: linear-gradient(135deg, #FF9A9E 0%, #FECFEF 100%);
}

.prediction-dog {
    background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 0.8rem;
    margin-top: 4rem;
    padding-top: 2rem;
    border-top: 1px solid rgba(255,255,255,0.1);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #00f2fe, #4facfe);
    color: #0E1117;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    box-shadow: 0 0 15px rgba(0, 242, 254, 0.5);
    transform: scale(1.02);
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
<div class="header-container">
    <h1 class="main-title">🧠 AI Image Classifier</h1>
    <p class="subtitle">Cat vs Dog Detection • Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING (OPTIMIZED & CACHED)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_images(base_folder, max_images=800):
    """Load and preprocess images with HOG feature extraction"""
    data = []
    labels = []
    
    categories = ["cats", "dogs"]
    
    for label, category in enumerate(categories):
        folder_path = os.path.join(base_folder, category)
        
        if not os.path.exists(folder_path):
            continue
            
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:max_images]
        
        for img_name in image_files:
            img_path = os.path.join(folder_path, img_name)
            
            try:
                # Read image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Resize to consistent size
                img = cv2.resize(img, (64, 64))
                
                # Extract HOG features
                features = hog(
                    img,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    visualize=False,
                    feature_vector=True
                )
                
                data.append(features)
                labels.append(label)
                
            except Exception as e:
                continue
    
    return np.array(data), np.array(labels)

# =============================================================================
# MODEL TRAINING (CACHED FOR PERFORMANCE)
# =============================================================================
@st.cache_resource(show_spinner=False)
def train_model():
    """Train SVM classifier with optimized parameters"""
    
    # Load training data
    X, y = load_images("training_set", max_images=800)
    
    if len(X) == 0:
        return None, 0, None, None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train SVM
    model = SVC(kernel="linear", probability=True, C=1.0)
    model.fit(X_train, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return model, accuracy, len(X_train), len(X_test)

# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================
def preprocess_image(image):
    """Preprocess uploaded image for prediction"""
    # Convert to grayscale
    if image.mode != 'L':
        img = np.array(image.convert("L"))
    else:
        img = np.array(image)
    
    # Resize to model input size
    img = cv2.resize(img, (64, 64))
    
    # Extract HOG features
    features = hog(
        img,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True
    )
    
    return features

# =============================================================================
# INITIALIZE MODEL
# =============================================================================
with st.spinner("🚀 Initializing AI model..."):
    start_time = time.time()
    model, accuracy, train_size, test_size = train_model()
    load_time = time.time() - start_time

if model is None:
    st.error("⚠️ Training data not found. Please ensure 'training_set/cats' and 'training_set/dogs' folders exist with images.")
    st.stop()

# =============================================================================
# DISPLAY MODEL METRICS
# =============================================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{accuracy * 100:.1f}%</p>
        <p class="metric-label">Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{train_size or 0}</p>
        <p class="metric-label">Training Images</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{load_time:.2f}s</p>
        <p class="metric-label">Load Time</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# IMAGE UPLOAD SECTION
# =============================================================================
st.markdown("### 📤 Upload Your Image")
st.markdown("Drag and drop or click to upload a cat or dog image")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# =============================================================================
# PREDICTION SECTION
# =============================================================================
if uploaded_file:
    # Load and display image
    image = Image.open(uploaded_file)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Add prediction button
    if st.button("🔍 Analyze Image", use_container_width=True):
        with st.spinner("🤖 Analyzing..."):
            # Simulate processing for better UX
            time.sleep(0.5)
            
            # Preprocess and predict
            features = preprocess_image(image)
            prediction = model.predict([features])[0]
            probabilities = model.predict_proba([features])[0]
            confidence = max(probabilities) * 100
            
            st.markdown("---")
            
            # Display prediction
            if prediction == 0:
                st.markdown("""
                <div class="prediction-cat">
                    🐱 PREDICTION: CAT
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-dog">
                    🐶 PREDICTION: DOG
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence visualization
            st.markdown(f"""
            <div class="confidence-container">
                <h4 style="margin-top:0;">Confidence Score</h4>
                <p style="font-size:1.5rem; font-weight:700; color:#667eea; margin:0;">
                    {confidence:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability breakdown
            st.markdown("#### 📊 Probability Breakdown")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("🐱 Cat", f"{probabilities[0]*100:.1f}%")
            with col2:
                st.metric("🐶 Dog", f"{probabilities[1]*100:.1f}%")
            
            # Progress bars for visual representation
            st.progress(probabilities[0], text=f"Cat: {probabilities[0]*100:.1f}%")
            st.progress(probabilities[1], text=f"Dog: {probabilities[1]*100:.1f}%")

else:
    # Show placeholder when no image uploaded
    st.info("👆 Upload an image above to get started!")
    
    # Sample instructions
    with st.expander("ℹ️ How to use this classifier"):
        st.markdown("""
        1. **Upload** a clear image of a cat or dog
        2. Click the **Analyze** button
        3. View the **prediction** and confidence score
        
        **Tips for best results:**
        - Use clear, well-lit images
        - Ensure the animal is the main subject
        - JPEG or PNG formats work best
        """)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div class="footer">
    <p>🚀 <strong>Machine Learning Internship Project</strong></p>
    <p>Prodigy InfoTech • Powered by SVM & HOG Features</p>
    <p style="font-size:0.8rem; margin-top:1rem;">
        Built with Streamlit • scikit-learn • OpenCV
    </p>
</div>
""", unsafe_allow_html=True)