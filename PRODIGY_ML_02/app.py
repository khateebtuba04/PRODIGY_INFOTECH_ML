import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="centered"
)

# ---------------- BASIC STYLING ----------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: white;
}
p, label {
    color: #cfd2d6;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🛍️ Customer Segmentation")
st.caption("K-Means clustering based on customer purchasing behavior")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Mall_Customers.csv")
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# ---------------- ELBOW METHOD ----------------
st.subheader("📉 Elbow Method")

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

fig1, ax1 = plt.subplots()
ax1.plot(range(1, 11), wcss, marker="o")
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("WCSS")
ax1.set_title("Optimal K")

st.pyplot(fig1)

# ---------------- SELECT K ----------------
k = st.slider("Select Number of Clusters", 2, 10, 5)

# ---------------- APPLY K-MEANS ----------------
kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

# ---------------- CLUSTER VISUALIZATION ----------------
st.subheader("📊 Customer Segments")

fig2, ax2 = plt.subplots()
ax2.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis"
)

ax2.set_xlabel("Annual Income (k$)")
ax2.set_ylabel("Spending Score (1–100)")
ax2.set_title("Customer Segmentation")

st.pyplot(fig2)
