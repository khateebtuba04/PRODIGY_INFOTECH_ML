# 🏠 House Price Prediction using Linear Regression

This project is developed as part of my **Machine Learning Internship at Prodigy Infotech**.

The goal of this task is to build a **Linear Regression model** to predict house prices based on key features and deploy it using an interactive web application.

---

## 📌 Task Objective
Implement a linear regression model to predict house prices based on:
- Square Footage
- Number of Bedrooms
- Number of Bathrooms

---

## 📂 Dataset
- **Source:** Kaggle – House Prices: Advanced Regression Techniques  
- **File Used:** `train.csv`

### Important Columns Used:
- `GrLivArea` – Above ground living area (in square feet)
- `BedroomAbvGr` – Number of bedrooms
- `FullBath` – Number of full bathrooms
- `SalePrice` – Target variable (house price)

The dataset contains historical housing data which is used to train and evaluate the model.

---

## 🧠 Machine Learning Model
- **Algorithm:** Linear Regression
- **Library:** Scikit-learn
- The dataset is split into training and testing sets.
- The model learns the relationship between house features and price.

---

## 🖥️ Web Application
The trained model is deployed using **Streamlit** to create a clean and interactive web interface.

### Features:
- User input for house details
- Real-time house price prediction
- Model performance visualization (Actual vs Predicted Prices)

---

## 📊 Visualization
- Scatter plot comparing actual house prices vs predicted prices
- Helps in understanding model performance and accuracy

---

## 🛠️ Tech Stack
- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib

---

## ▶️ How to Run the Project

1. Install required libraries:
```bash
pip install streamlit pandas scikit-learn matplotlib
```

2.Run the application:
```bash
streamlit run app.py
```

3.Open your browser at:
```bash
http://localhost:8501
```
