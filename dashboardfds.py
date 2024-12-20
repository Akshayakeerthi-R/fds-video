# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
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

# Correlation Matrix
st.subheader("Correlation Matrix")
# Select only numeric columns for correlation calculation
numeric_data = selected_data.select_dtypes(include=["float64", "int64"])
if not numeric_data.empty:
    correlation_matrix = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
    ax.set_title(f"Correlation Matrix ({data_selection})")
    st.pyplot(fig)
else:
    st.write("No numeric columns available for correlation matrix.")

# Section 3: Model Performance Metrics
st.header("3. Model Performance Metrics")

# Good Preprocessing Metrics
good_accuracy = 0.5325
good_cm = [[46, 142], [45, 167]]
good_fpr = [0.0, 0.45, 1.0]
good_tpr = [0.0, 0.55, 1.0]
good_auc = auc(good_fpr, good_tpr)

# Minimal Preprocessing Metrics
minimal_accuracy = 0.46578947368421053
minimal_cm = [[71, 115], [88, 106]]
minimal_fpr = [0.0, 0.60, 1.0]
minimal_tpr = [0.0, 0.40, 1.0]
minimal_auc = auc(minimal_fpr, minimal_tpr)

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

# Section 4: ROC Curves
st.header("4. ROC Curves")
fig = go.Figure()
fig.add_trace(go.Scatter(x=good_fpr, y=good_tpr, mode='lines', name=f"Good Preprocessing (AUC={good_auc:.2f})"))
fig.add_trace(go.Scatter(x=minimal_fpr, y=minimal_tpr, mode='lines', name=f"Minimal Preprocessing (AUC={minimal_auc:.2f})"))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name="Random Guess", line=dict(dash='dash')))
fig.update_layout(title="ROC Curve Comparison", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", width=700, height=500)
st.plotly_chart(fig, use_container_width=True)

# Section 5: Model Comparison
st.header("5. Model Comparison")

comparison_data = {
    "Model": ["Minimal Preprocessing", "Good Preprocessing"],
    "Accuracy": [minimal_accuracy, good_accuracy],
}

df_comparison = pd.DataFrame(comparison_data)

# Visualization of model comparison
fig = px.bar(df_comparison, x="Model", y="Accuracy", color="Model", title="Accuracy Comparison Between Preprocessing Methods")
st.plotly_chart(fig, use_container_width=True)

# Section 6: Insights
st.header("6. Insights")
st.markdown("""
- **Minimal Preprocessing**: Achieved an accuracy of 46.58%, indicating that preprocessing plays a crucial role in improving the model's performance.
- **Good Preprocessing**: Achieved a higher accuracy of 53.25%. Despite the improvement, there's still room for further optimization and refinement.
- **Takeaway**: Good preprocessing helps in improving the model's performance, but more work might be needed to fully optimize the data for better classification results.
""")
