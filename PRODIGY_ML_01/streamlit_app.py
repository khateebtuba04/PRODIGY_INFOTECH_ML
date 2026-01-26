import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ---------------- PAGE SETTINGS ---------------- #
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.write("Predict house prices using Linear Regression")

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv("train.csv")

X = df[["GrLivArea", "BedroomAbvGr", "FullBath"]]
y = df["SalePrice"]

X = X.fillna(X.mean())

# ---------------- TRAIN MODEL ---------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------- USER INPUT ---------------- #
st.subheader("Enter House Details")

area = st.number_input("Square Footage", min_value=300, step=10)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, step=1)

# ---------------- PREDICT ---------------- #
if st.button("Predict Price"):
    input_data = pd.DataFrame(
        [[area, bedrooms, bathrooms]],
        columns=["GrLivArea", "BedroomAbvGr", "FullBath"]
    )

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated House Price: ₹ {int(prediction):,}")

    # ---------------- GRAPH ---------------- #
    y_pred = model.predict(X_test)

    st.subheader("📊 Model Performance")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")
    ax.set_title("Actual vs Predicted Prices")

    st.pyplot(fig)
