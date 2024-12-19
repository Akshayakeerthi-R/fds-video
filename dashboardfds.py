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

# Load the data
raw_data_path = "https://raw.githubusercontent.com/Akshayakeerthi-R/fds-video/refs/heads/main/diabetes_prediction_dataset.csv"
df_raw = pd.read_csv(raw_data_path)

# Minimal Preprocessing: Handling missing values and scaling
def minimal_preprocessing(df):
    # Handle missing values for numeric features with SimpleImputer (using median strategy)
    numeric_features = df.select_dtypes(include=[np.number]).columns
    numeric_imputer = SimpleImputer(strategy='median')
    df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])
    
    # No encoding for categorical features (i.e., not applying one-hot encoding)
    # No scaling applied in minimal preprocessing
    return df

# Good Preprocessing: Handling missing values, one-hot encoding, and scaling
def good_preprocessing(df):
    # Handle missing values: Fill missing values in numeric columns with the median
    numeric_features = df.select_dtypes(include=[np.number]).columns
    numeric_imputer = SimpleImputer(strategy='median')
    df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])

    # Handle categorical variables with One-Hot Encoding
    categorical_features = ["Smoking_Status", "Alcohol_Consumption"]
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # Feature scaling: Standard Scaling for numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop(columns=["Outcome", "Patient_ID"]))
    
    # Returning the scaled DataFrame
    X_scaled = pd.DataFrame(X_scaled, columns=df.drop(columns=["Outcome", "Patient_ID"]).columns)
    df = pd.concat([X_scaled, df[["Outcome"]]], axis=1)
    
    return df

# Section 1: Data Overview
st.header("1. Data Overview")
st.markdown("""
This application compares the performance of models trained with minimal preprocessing versus good preprocessing on a diabetes prediction dataset.
""")

# Section 2: Data Distribution Visualization
st.header("2. Data Distribution Visualization")

# Visualizing Data Distribution
data_selection = st.sidebar.radio("Choose data to display:", ("Raw Data", "Minimal Preprocessed Data", "Good Preprocessed Data"))
df_minimal_preprocessed = minimal_preprocessing(df_raw.copy())
df_good_preprocessed = good_preprocessing(df_raw.copy())

if data_selection == "Raw Data":
    selected_data = df_raw
elif data_selection == "Minimal Preprocessed Data":
    selected_data = df_minimal_preprocessed
else:
    selected_data = df_good_preprocessed

st.write(f"### {data_selection} Distribution")

# Iterate through columns and plot the distribution
for col in selected_data.select_dtypes(include=["object", "int64", "float64"]).columns:
    # Plot histogram for numeric columns
    if selected_data[col].dtype in ["int64", "float64"]:
        fig = px.histogram(selected_data, x=col, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
    
    # For categorical features, plot a bar chart
    elif selected_data[col].dtype == "object":
        fig = px.bar(selected_data[col].value_counts().reset_index(), 
                     x="index", y=col, 
                     labels={"index": col, col: "Count"},
                     title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)

# Section 3: Model Performance Metrics
st.header("3. Model Performance Metrics")

# Minimal Preprocessing Metrics
minimal_accuracy = 0.46578947368421053
minimal_cm = [[71, 115], [88, 106]]

# Display Minimal Preprocessing metrics
st.write("### Minimal Preprocessing Metrics")
st.write(f"**Accuracy:** {minimal_accuracy}")
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(minimal_cm, annot=True, fmt='d', cmap="Blues", cbar=False, xticklabels=["Non-Diabetic", "Diabetic"], yticklabels=["Non-Diabetic", "Diabetic"])
ax.set_title("Confusion Matrix (Minimal Preprocessing)")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
st.pyplot(fig)

# Good Preprocessing Metrics
good_accuracy = 0.5325
good_cm = [[46, 142], [45, 167]]

# Display Good Preprocessing metrics
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
- **Minimal Preprocessing**: Achieved an accuracy of 46.58%, which indicates that preprocessing steps significantly improve model performance.
- **Good Preprocessing**: Achieved an accuracy of 53.25%. The confusion matrix suggests that while the model is decent, it still struggles with some classifications.
- **Takeaway**: Proper preprocessing can enhance the model's ability to generalize, reducing misclassifications.
""")
