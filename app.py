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
selected_features = ['n_tokens_title', 'num_imgs', 'num_videos', 'num_keywords']
X = df[selected_features]

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
# -------------------------------
# ✅ Predict News Popularity Section
# -------------------------------
st.subheader("🧠 Predict News Popularity from Your Input")
st.markdown("Enter values for each feature below:")

# Dynamically generate input fields for all model features
user_input = {}
for feature in X.columns:
    col_data = df[feature]
    min_val = float(col_data.min())
    max_val = float(col_data.max())
    mean_val = float(col_data.mean())

    # Use slider for better user experience
    if col_data.dtype == 'float64':
        user_input[feature] = st.slider(f"{feature}", min_value=min_val, max_value=max_val, value=mean_val)
    else:
        user_input[feature] = st.slider(f"{feature}", min_value=int(min_val), max_value=int(max_val), value=int(mean_val))

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Predict when button is clicked
if st.button("🔍 Predict"):
    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        prob = model.predict_proba(input_scaled)[0][1]

        if prediction[0] == 1:
            st.success(f"✅ Your article is likely to be POPULAR! (Confidence: {prob:.2%})")
        else:
            st.warning(f"❌ Your article is likely NOT popular. (Confidence: {(1 - prob):.2%})")
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {str(e)}")
