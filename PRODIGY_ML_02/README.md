# 🛍️ Customer Segmentation using K-Means Clustering

This project is developed as part of my **Machine Learning Internship at Prodigy Infotech**.

The goal of this task is to segment retail customers into distinct groups based on their purchasing behavior using **K-Means clustering**.

---

## 📌 Task Objective
To group customers of a retail store based on:
- Annual Income
- Spending Score

This helps businesses understand different customer types and design targeted marketing strategies.

---

## 📂 Dataset
- **Source:** Kaggle – Customer Segmentation Tutorial in Python  
- **File Used:** `Mall_Customers.csv`

### Features Used:
- `Annual Income (k$)`
- `Spending Score (1-100)`

---

## 🧠 Machine Learning Model
- **Algorithm:** K-Means Clustering
- **Type:** Unsupervised Learning
- **Cluster Selection:** Elbow Method

The Elbow Method is used to determine the optimal number of customer segments.

---

## 📊 Visualizations
- Elbow Method graph to identify optimal clusters
- Scatter plot showing customer segments based on income and spending behavior

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
3.Open browser at:
```bash
http://localhost:8501
```
```bash
pip install streamlit pandas scikit-learn matplotlib
