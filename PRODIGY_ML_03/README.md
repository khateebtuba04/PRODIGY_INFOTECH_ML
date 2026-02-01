``` markdown
# 🧠 Cat vs Dog Image Classification (Task 03)

This project is part of my **Machine Learning Internship at Prodigy Infotech**.  
The objective of this task is to build an image classification system that can identify whether an uploaded image is a **Cat 🐱 or a Dog 🐶**.

---

## 📌 Project Objective

- To classify images as **Cat or Dog**
- To apply **classical machine learning techniques** for image classification
- To deploy the model using an interactive **Streamlit web application**

---

## 🧠 Approach Used

- **Algorithm:** Support Vector Machine (SVM)
- **Feature Extraction:** Histogram of Oriented Gradients (HOG)
- **Image Processing:** OpenCV
- **Web Interface:** Streamlit

This project demonstrates the **complete ML workflow**, from data preprocessing to model deployment.

---

## 📂 Dataset

- Dataset: **Cats vs Dogs Dataset (Kaggle)**
- Images are organized into two folders:
```

training_set/
├── cats/
└── dogs/

````

⚠️ The dataset is **not included** in this repository due to size limitations.

---

## ⚙️ How the Model Works

1. Image is converted to grayscale  
2. Image is resized to a fixed size  
3. HOG features are extracted  
4. SVM model is trained on extracted features  
5. User uploads an image  
6. Model predicts **Cat or Dog**  
7. Prediction confidence is displayed  

---

## 🖥️ Application Features

- Clean and simple UI
- Image upload functionality
- Real-time prediction
- Confidence score display
- Optimized performance using caching

---

## 🚀 How to Run the Project

### 1️⃣ Install Required Libraries
```bash
pip install streamlit scikit-learn opencv-python scikit-image pillow
````

### 2️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 📊 Model Performance

* The model provides **reasonable accuracy** for a classical ML approach
* Some misclassifications may occur due to:

  * Background noise
  * Lighting conditions
  * Similar visual patterns between cats and dogs

This behavior is **expected** and acceptable for this task.

---

## 📌 Note

This project focuses on **classical machine learning** techniques.
For higher accuracy, deep learning models such as CNNs would be more suitable, but they are beyond the scope of this task.

---

## 🏁 Conclusion

Through this task, I gained hands-on experience in:

* Image preprocessing
* Feature extraction
* Model training and evaluation
* Deploying ML models using Streamlit

---

## 👩‍💻 Internship Details

* **Internship Role:** Machine Learning Intern
* **Organization:** Prodigy Infotech
* **Task:** Task 03 – Image Classification

---

✨ *Thank you for reviewing this project!* ✨

```

