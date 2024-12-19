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

# Section 2: Model Performance Metrics
st.header("2. Model Performance Metrics")

# Good Preprocessing Metrics
good_accuracy = 0.5325
good_cm = [[46, 142], [45, 167]]

# Minimal Preprocessing Metrics
minimal_accuracy = 0.46578947368421053
minimal_cm = [[71, 115], [88, 106]]

# Display Good Preprocessing metrics
st.write("### Good Preprocessing Metrics")
st.write(f"**Accuracy:** {good_accuracy}")
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(good_cm, annot=True, fmt='d', cmap="Blues", cbar=False, xticklabels=["Non-Diabetic", "Diabetic"], yticklabels=["Non-Diabetic", "Diabetic"])
ax.set_title("Confusion Matrix (Good Preprocessing)")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
st.pyplot(fig)

# Display Minimal Preprocessing metrics
st.write("### Minimal Preprocessing Metrics")
st.write(f"**Accuracy:** {minimal_accuracy}")
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(minimal_cm, annot=True, fmt='d', cmap="Blues", cbar=False, xticklabels=["Non-Diabetic", "Diabetic"], yticklabels=["Non-Diabetic", "Diabetic"])
ax.set_title("Confusion Matrix (Minimal Preprocessing)")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
st.pyplot(fig)

# Section 3: Model Comparison
st.header("3. Model Comparison")

comparison_data = {
    "Model": ["Good Preprocessing", "Minimal Preprocessing"],
    "Accuracy": [good_accuracy, minimal_accuracy],
}

df_comparison = pd.DataFrame(comparison_data)

# Visualization of model comparison
fig = px.bar(df_comparison, x="Model", y="Accuracy", color="Model", title="Accuracy Comparison Between Preprocessing Methods")
st.plotly_chart(fig, use_container_width=True)

# Section 4: Insights
st.header("4. Insights")
st.markdown("""
- **Good Preprocessing**: Achieved an accuracy of 53.25%. The confusion matrix suggests that while the model is decent, it still struggles with some classifications.
- **Minimal Preprocessing**: Lower accuracy of 46.58%, which indicates that preprocessing steps significantly improve model performance.
- **Takeaway**: Proper preprocessing can enhance the model's ability to generalize, reducing misclassifications.
""")
