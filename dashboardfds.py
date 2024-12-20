import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go

# Title of the Streamlit App
st.title("Model Comparison: Minimal vs Good Preprocessing")

st.sidebar.title("Options")

# Load your datasets or any data needed
raw_data_path = "https://raw.githubusercontent.com/Akshayakeerthi-R/fds-video/refs/heads/main/diabetes_prediction_dataset.csv"
preprocessed_data_path = "https://raw.githubusercontent.com/Akshayakeerthi-R/fds-video/refs/heads/main/preprocessed_dataset.csv"

# Load the data
try:
    df_raw = pd.read_csv(raw_data_path)
    df_preprocessed = pd.read_csv(preprocessed_data_path)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Good Preprocessing Metrics
good_accuracy = 0.5325
good_cm = [[46, 142], [45, 167]]

# Minimal Preprocessing Metrics
minimal_accuracy = 0.46578947368421053
minimal_cm = [[71, 115], [88, 106]]

# Load true labels and predicted probabilities
# Replace with actual data sources
y_true_minimal = pd.read_csv("y_true_minimal.csv").values.ravel()
y_score_minimal = pd.read_csv("y_score_minimal.csv").values.ravel()

y_true_good = pd.read_csv("y_true_good.csv").values.ravel()
y_score_good = pd.read_csv("y_score_good.csv").values.ravel()

# Section 1: Data Overview
st.header("1. Data Overview")
st.markdown("""
This application compares the performance of models trained with minimal preprocessing versus good preprocessing on a diabetes prediction dataset.
""")

# Section 2: Data Distribution Visualization
st.header("2. Data Distribution Visualization")
data_selection = st.sidebar.radio("Choose data to display:", ("Raw Data", "Preprocessed Data"))
selected_data = df_raw if data_selection == "Raw Data" else df_preprocessed

st.write(f"### {data_selection} Distribution")
for col in selected_data.select_dtypes(include=["object", "int64", "float64"]).columns:
    fig = px.histogram(selected_data, x=col, title=f"Distribution of {col}")
    st.plotly_chart(fig, use_container_width=True)

# Section 3: Correlation Matrix
st.header("3. Correlation Matrix")
option = st.selectbox("Choose Preprocessing Type:", ["Minimal Preprocessing", "Good Preprocessing"])

if option == "Minimal Preprocessing":
    st.subheader("Correlation Matrix: Minimal Preprocessing")
    corr_matrix_minimal = df_raw.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix_minimal, annot=True, cmap="coolwarm")
    st.pyplot(plt)

elif option == "Good Preprocessing":
    st.subheader("Correlation Matrix: Good Preprocessing")
    corr_matrix_good = df_preprocessed.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix_good, annot=True, cmap="coolwarm")
    st.pyplot(plt)

# Section 4: Model Performance Metrics
st.header("4. Model Performance Metrics")

# Display Minimal Preprocessing metrics
st.write("### Minimal Preprocessing Metrics")
st.write(f"**Accuracy:** {minimal_accuracy}")
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(minimal_cm, annot=True, fmt='d', cmap="Blues", cbar=False, xticklabels=["Non-Diabetic", "Diabetic"], yticklabels=["Non-Diabetic", "Diabetic"])
ax.set_title("Confusion Matrix (Minimal Preprocessing)")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
st.pyplot(fig)

st.subheader("ROC Curve: Minimal Preprocessing")
fpr, tpr, _ = roc_curve(y_true_minimal, y_score_minimal)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Minimal Preprocessing')
plt.legend(loc="lower right")
st.pyplot(plt)

# Display Good Preprocessing metrics
st.write("### Good Preprocessing Metrics")
st.write(f"**Accuracy:** {good_accuracy}")
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(good_cm, annot=True, fmt='d', cmap="Blues", cbar=False, xticklabels=["Non-Diabetic", "Diabetic"], yticklabels=["Non-Diabetic", "Diabetic"])
ax.set_title("Confusion Matrix (Good Preprocessing)")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
st.pyplot(fig)

st.subheader("ROC Curve: Good Preprocessing")
fpr, tpr, _ = roc_curve(y_true_good, y_score_good)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Good Preprocessing')
plt.legend(loc="lower right")
st.pyplot(plt)

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
