import cv2
import numpy as np
import tensorflow as tf
import json
import os

def load_model_and_labels():
    # Get project root directory (parent of src)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    model_path = os.path.join(project_root, 'hand_gesture_model.h5')
    json_path = os.path.join(project_root, 'class_indices.json')

    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please train the model first.")
        return None, None
    
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            class_indices = json.load(f)
        # Invert the dictionary to map index -> class name
        labels = {v: k for k, v in class_indices.items()}
        print("Class labels loaded.")
    else:
        # Fallback if no json found, though train.py generates it
        print("Warning: class_indices.json not found. Using default labels.")
        labels = {i: str(i) for i in range(20)}
        
    return model, labels

def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize to match model input
    resized = cv2.resize(gray, (128, 128))
    # Normalize
    normalized = resized / 255.0
    # Reshape for model (1, 128, 128, 1)
    reshaped = normalized.reshape(1, 128, 128, 1)
    return reshaped

def main():
    model, labels = load_model_and_labels()
    if model is None:
        return

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Define a region of interest (ROI) box
    roi_top, roi_bottom = 50, 350
    roi_left, roi_right = 50, 350
    
    # Flag to invert the binary mask (useful if background is light/dark)
    invert_mode = False

    print("Controls:")
    print("  'q' - Quit")
    print("  'i' - Invert binary mask colors (Toggle)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Draw ROI on frame
        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)

        # Extract ROI
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]
        
        # --- Preprocessing Step ---
        # 1. Convert to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply Gaussian Blur to reduce noise
        gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
        
        # 3. Apply Thresholding to create a binary mask (Black/White)
        # We use Otsu's binarization which automatically finds the threshold
        threshold_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU if invert_mode else cv2.THRESH_BINARY + cv2.THRESH_OTSU
        ret_val, binary_roi = cv2.threshold(gray_roi, 0, 255, threshold_type)
        
        # 4. Prepare for Model
        # Resize to 128x128
        resized = cv2.resize(binary_roi, (128, 128))
        # Normalize to [0, 1]
        normalized = resized / 255.0
        # Reshape to (1, 128, 128, 1)
        input_data = normalized.reshape(1, 128, 128, 1)
        
        # Predict
        try:
            prediction = model.predict(input_data, verbose=0)
            class_idx = np.argmax(prediction)
            confidence = np.max(prediction)
            
            label_text = labels.get(class_idx, "Unknown")
            
            # Display Prediction
            text = f"Gesture: {label_text} ({confidence*100:.1f}%)"
            color = (0, 255, 0) if confidence > 0.7 else (0, 0, 255)
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, color, 2, cv2.LINE_AA)
            
        except Exception as e:
            print(f"Prediction error: {e}")

        # Display Main Frame
        cv2.imshow('Hand Gesture Recognition', frame)
        
        # Display "What the model sees"
        cv2.imshow('Model Input (ROI)', binary_roi)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('i'):
            invert_mode = not invert_mode
            print(f"Invert Mode: {invert_mode}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
