# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
import plotly.graph_objects as go

# Title and Sidebar
st.title("Model Comparison: Minimal vs Good Preprocessing")
st.sidebar.title("Options")

# Section 1: Data Overview
st.header("1. Data Overview")
st.markdown("""
This application compares the performance of models trained with minimal preprocessing versus good preprocessing on a diabetes prediction dataset.
""")

# Load your datasets or any data needed
raw_data_path = "https://raw.githubusercontent.com/Akshayakeerthi-R/fds-video/refs/heads/main/diabetes_prediction_dataset.csv"
# Assume that preprocessed data is available and accessible
preprocessed_data_path = "https://raw.githubusercontent.com/Akshayakeerthi-R/fds-video/refs/heads/main/preprocessed_dataset.csv"

# Load the data
try:
    df_raw = pd.read_csv(raw_data_path)
    df_preprocessed = pd.read_csv(preprocessed_data_path)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Section 2: Data Distribution Visualization
st.header("2. Data Distribution Visualization")

# Data distribution for Raw and Preprocessed data
data_selection = st.sidebar.radio("Choose data to display:", ("Raw Data", "Preprocessed Data"))
selected_data = df_raw if data_selection == "Raw Data" else df_preprocessed

st.write(f"### {data_selection} Distribution")
for col in selected_data.select_dtypes(include=["object", "int64", "float64"]).columns:
    fig = px.histogram(selected_data, x=col, title=f"Distribution of {col}")
    st.plotly_chart(fig, use_container_width=True)

# Section 3: Model Performance Metrics
st.header("3. Model Performance Metrics")

# Good Preprocessing Metrics
good_accuracy = 0.5325
good_cm = [[46, 142], [45, 167]]

# Minimal Preprocessing Metrics
minimal_accuracy = 0.46578947368421053
minimal_cm = [[71, 115], [88, 106]]

# Display Minimal Preprocessing metrics first
st.write("### Minimal Preprocessing Metrics")
st.write(f"**Accuracy:** {minimal_accuracy}")
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(minimal_cm, annot=True, fmt='d', cmap="Blues", cbar=False, xticklabels=["Non-Diabetic", "Diabetic"], yticklabels=["Non-Diabetic", "Diabetic"])
ax.set_title("Confusion Matrix (Minimal Preprocessing)")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
st.pyplot(fig)

# Display Good Preprocessing metrics later
st.write("### Good Preprocessing Metrics")
st.write(f"**Accuracy:** {good_accuracy}")
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(good_cm, annot=True, fmt='d', cmap="Blues", cbar=False, xticklabels=["Non-Diabetic", "Diabetic"], yticklabels=["Non-Diabetic", "Diabetic"])
ax.set_title("Confusion Matrix (Good Preprocessing)")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
st.pyplot(fig)

# Section 4: Model Comparison
st.header("4. Model Comparison")

comparison_data = {
    "Model": ["Minimal Preprocessing", "Good Preprocessing"],
    "Accuracy": [minimal_accuracy, good_accuracy],
}

df_comparison = pd.DataFrame(comparison_data)

# Visualization of model comparison
fig = px.bar(df_comparison, x="Model", y="Accuracy", color="Model", title="Accuracy Comparison Between Preprocessing Methods")
st.plotly_chart(fig, use_container_width=True)

# Section 5: Insights
st.header("5. Insights")
st.markdown("""
- **Minimal Preprocessing**: Achieved an accuracy of 46.58%, indicating that preprocessing plays a crucial role in improving the model's performance.
- **Good Preprocessing**: Achieved a higher accuracy of 53.25%. Despite the improvement, there's still room for further optimization and refinement.
- **Takeaway**: Good preprocessing helps in improving the model's performance, but more work might be needed to fully optimize the data for better classification results.
""")
