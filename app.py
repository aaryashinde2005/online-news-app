# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Title
st.title("📰 Online News Popularity Prediction App")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('OnlineNewsPopularity.csv')


    df.columns = df.columns.str.strip()
    df.drop(['url', 'timedelta'], axis=1, inplace=True)
    df['popular'] = (df['shares'] > 1400).astype(int)
    df.drop('shares', axis=1, inplace=True)
    return df

df = load_data()
st.subheader("📊 Dataset Preview")
st.write(df.head())

# Show class distribution
st.subheader("🧮 Popular vs Not Popular Distribution")
st.bar_chart(df['popular'].value_counts())

# Split features/labels
X = df.drop('popular', axis=1)
y = df['popular']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Metrics
st.subheader("📋 Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Confusion Matrix
st.subheader("🔍 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Popular", "Popular"], yticklabels=["Not Popular", "Popular"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)
