# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go

# Load your datasets or any data needed
raw_data_path = "https://raw.githubusercontent.com/Iamvideo123/fds/refs/heads/main/diabetes_prediction_dataset.csv"
# Assume that preprocessed data is available and accessible
preprocessed_data_path = "https://raw.githubusercontent.com/Iamvideo123/fds/refs/heads/main/diabetes_prediction_dataset_preprocessed.csv"

# Load the data
try:
    df_raw = pd.read_csv(raw_data_path)
    df_preprocessed = pd.read_csv(preprocessed_data_path)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Title and Sidebar
st.title("Model Comparison: Minimal vs Good Preprocessing")
st.sidebar.title("Options")

# Section 1: Data Distribution Visualization
st.header("1. Data Distribution Visualization")

data_selection = st.sidebar.radio("Choose data to display:", ("Raw Data", "Preprocessed Data"))
selected_data = df_raw if data_selection == "Raw Data" else df_preprocessed

st.write(f"### {data_selection} Distribution")
for col in selected_data.select_dtypes(include=["object", "int64", "float64"]).columns:
    fig = px.histogram(selected_data, x=col, title=f"Distribution of {col}")
    st.plotly_chart(fig, use_container_width=True)

# Section 2: Model Performance Metrics
st.header("2. Model Performance Metrics")

# Classification report for raw data
raw_metrics = {
    "Accuracy": 0.9975,
    "Precision": 1.0,
    "Recall": 1.0,
    "F1-Score": 1.0,
}

# Classification report for preprocessed data
preprocessed_metrics = {
    "Accuracy": 0.86375,
    "Precision": 0.8641,
    "Recall": 0.8638,
    "F1-Score": 0.8636,
}

# Model metrics for raw data
raw_report = {
    "Class 0": {"Precision": 1.0, "Recall": 1.0, "F1-Score": 1.0},
    "Class 1": {"Precision": 1.0, "Recall": 1.0, "F1-Score": 1.0},
}

# Model metrics for preprocessed data
preprocessed_report = {
    "Class 0": {"Precision": 0.8555, "Recall": 0.8902, "F1-Score": 0.8725},
    "Class 1": {"Precision": 0.8736, "Recall": 0.8346, "F1-Score": 0.8537},
}

# Display metrics for raw data
st.write("### Training Metrics (Raw Data)")
raw_df_metrics = pd.DataFrame(raw_report).T
fig = px.bar(raw_df_metrics, title="Classification Metrics for Raw Data")
st.plotly_chart(fig, use_container_width=True)

# Display metrics for preprocessed data
st.write("### Training Metrics (Preprocessed Data)")
preprocessed_df_metrics = pd.DataFrame(preprocessed_report).T
fig = px.bar(preprocessed_df_metrics, title="Classification Metrics for Preprocessed Data")
st.plotly_chart(fig, use_container_width=True)

# Section 3: Model Comparison
st.header("3. Model Comparison")

# Model comparison table
comparison_data = {
    "Model": ["Raw Data", "Preprocessed Data"],
    "Accuracy": [raw_metrics["Accuracy"], preprocessed_metrics["Accuracy"]],
    "Precision": [raw_metrics["Precision"], preprocessed_metrics["Precision"]],
    "Recall": [raw_metrics["Recall"], preprocessed_metrics["Recall"]],
    "F1-Score": [raw_metrics["F1-Score"], preprocessed_metrics["F1-Score"]],
}
df_comparison = pd.DataFrame(comparison_data)

# Visualization of model comparison
fig = px.bar(df_comparison, x="Model", y=["Accuracy", "Precision", "Recall", "F1-Score"],
             barmode="group", title="Performance Comparison Between Raw and Preprocessed Data")
st.plotly_chart(fig, use_container_width=True)

# Display comparison table
st.write("### Comparison Table")
st.dataframe(df_comparison)

# Section 4: Confusion Matrix and ROC Curve
st.header("4. Confusion Matrix and ROC Curve")

# Dummy confusion matrices for raw and preprocessed data
cm_raw = confusion_matrix([0, 1, 0, 1, 1], [0, 1, 0, 1, 0])  # Example
cm_preprocessed = confusion_matrix([0, 1, 0, 1, 1], [0, 1, 0, 1, 1])  # Example

# Plot confusion matrix for Raw Data
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.heatmap(cm_raw, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Non-Diabetic", "Diabetic"], yticklabels=["Non-Diabetic", "Diabetic"], ax=axes[0])
axes[0].set_title("Confusion Matrix (Raw Data)")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

# Plot confusion matrix for Preprocessed Data
sns.heatmap(cm_preprocessed, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Non-Diabetic", "Diabetic"], yticklabels=["Non-Diabetic", "Diabetic"], ax=axes[1])
axes[1].set_title("Confusion Matrix (Preprocessed Data)")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")

st.pyplot(fig)

# Dummy ROC curve data for raw and preprocessed models
fpr_raw, tpr_raw, _ = roc_curve([0, 1, 0, 1, 1], [0.2, 0.8, 0.1, 0.9, 0.6])
fpr_preprocessed, tpr_preprocessed, _ = roc_curve([0, 1, 0, 1, 1], [0.1, 0.7, 0.05, 0.85, 0.65])

# Plot ROC curves
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr_raw, y=tpr_raw, mode='lines', name='Raw Data ROC Curve'))
fig_roc.add_trace(go.Scatter(x=fpr_preprocessed, y=tpr_preprocessed, mode='lines', name='Preprocessed Data ROC Curve'))
fig_roc.update_layout(title="ROC Curve Comparison", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
st.plotly_chart(fig_roc, use_container_width=True)

# Section 5: Insights
st.header("5. Insights")
st.markdown("""
- **Raw Data**: The model performs exceptionally well with an accuracy of 99.75%, but this could indicate overfitting.
- **Preprocessed Data**: After preprocessing, the accuracy drops to 86.38%, with more realistic precision, recall, and F1-score values.
- **Significance**: Preprocessing reduces overfitting and results in a more generalizable model.
""")
