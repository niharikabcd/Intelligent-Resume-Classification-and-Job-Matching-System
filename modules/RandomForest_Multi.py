import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from datasets import load_dataset
import joblib
import os

# Load the dataset
ds = load_dataset('ahmedheakl/resume-atlas', cache_dir="C:/Users/dell/.cache/huggingface/datasets")

# Load the dataset and create a DataFrame from the 'train' split
df_train = pd.DataFrame(ds['train'])

# Initialize the Label Encoder
le = LabelEncoder()
df_train['Category_encoded'] = le.fit_transform(df_train['Category'])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df_train['Text'], df_train['Category_encoded'], test_size=0.2, random_state=42)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Define the model file path for saving/loading
model_file = 'random_forest_multi_model.pkl'

# Check if the model file exists
if os.path.exists(model_file):
    # Load the model if it exists
    rf_multi = joblib.load(model_file)
else:
    # Initialize and train the Random Forest model
    rf_multi = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_multi.fit(X_train_tfidf, y_train)
    
    # Save the trained model
    joblib.dump(rf_multi, model_file)

# Multi-label classification (returning top N predictions based on probabilities)
def classify_text_rf_multi(text, top_n=3):
    text_tfidf = vectorizer.transform([text])
    probabilities = rf_multi.predict_proba(text_tfidf)[0]
    top_n_indices = np.argsort(probabilities)[::-1][:top_n]  # Get indices of top N predictions
    top_n_categories = le.inverse_transform(top_n_indices)
    return top_n_categories
