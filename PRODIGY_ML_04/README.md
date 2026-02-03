```md
# Hand Gesture Recognition System 🖐️  
**Prodigy Infotech – Task 04**

## 📌 Project Overview

This project implements a **Hand Gesture Recognition System** using a **Convolutional Neural Network (CNN)**.  
The system is capable of recognizing and classifying different hand gestures from images and performing **real-time gesture recognition using a webcam**.

The project demonstrates the use of:
- Deep Learning (CNN)
- Image preprocessing
- Real-time inference with OpenCV
- Structured and modular Python code

---
## 🎯 Objective

To develop a robust hand gesture recognition model that:
- Learns gesture patterns from image data
- Accurately classifies multiple hand gestures
- Enables intuitive human–computer interaction
- Performs real-time predictions using a webcam

---

## 📂 Dataset

- Hand Gesture Recognition Dataset (Kaggle / Alternate Dataset)
- Images are organized into class-wise folders
- Dataset is **not uploaded** to this repository due to size limitations

📌 You can use any hand gesture dataset with a similar folder structure.

---

## 🛠️ Technologies Used

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **JSON**

---

## 🗂️ Project Structure

```

Hand-Gesture-Recognition/
│
├── src/
│   ├── train.py        # Model training script
│   ├── model.py        # CNN architecture
│   └── inference.py    # Real-time webcam inference
│
├── hand_gesture_model.h5      # Trained model file
├── class_indices.json         # Class index to label mapping
├── requirements.txt           # Required Python libraries
├── .gitignore
└── README.md

````

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Hand-Gesture-Recognition.git
cd Hand-Gesture-Recognition
````

---

### 2️⃣ Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Mac / Linux**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Training the Model (Optional)

If you want to train the model from scratch:

```bash
python src/train.py
```

This will:

* Load the dataset
* Train the CNN model
* Save the trained model as `hand_gesture_model.h5`
* Save class label mappings in `class_indices.json`

📌 If the model file already exists, you can skip this step.

---

## 🎥 Running Real-Time Hand Gesture Recognition

To start the webcam-based gesture recognition system, run:

```bash
python src/inference.py
```

---

## 🎮 Controls During Inference

| Key   | Action                       |
| ----- | ---------------------------- |
| **q** | Quit the application         |
| **i** | Toggle binary mask inversion |

📌 To **exit the program**, press **`q`** on the keyboard.

---

## 🧪 How It Works (Inference Pipeline)

1. Webcam captures video frames
2. Region of Interest (ROI) is extracted
3. Frame is converted to grayscale
4. Gaussian blur and thresholding are applied
5. Image is resized and normalized
6. CNN model predicts the gesture
7. Gesture label and confidence are displayed in real time

---

## 📊 Output

* Displays detected gesture name
* Shows confidence score
* Visualizes ROI and processed input
* Runs smoothly in real time using a webcam

---

## 🚀 Future Improvements

* Improve accuracy with more training data
* Add support for dynamic gestures
* Integrate gesture-based application control
* Deploy as a desktop or web application

---

## 🏢 Internship Information

**Internship:** Prodigy Infotech
**Domain:** Machine Learning
**Task:** Task 04 – Hand Gesture Recognition System

---

## ✅ Conclusion

This project demonstrates the complete pipeline of a hand gesture recognition system, from data preprocessing and CNN-based training to real-time webcam inference. It highlights practical skills in deep learning, computer vision, and software structuring suitable for real-world applications.

---
