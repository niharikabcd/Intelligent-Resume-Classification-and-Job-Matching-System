import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
import joblib
import os

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

# Initialize and train the Random Forest model
'''rf_single = RandomForestClassifier(n_estimators=100, random_state=42)
rf_single.fit(X_train_tfidf, y_train)'''

# Define the model file path for saving/loading
model_file = 'random_forest_model.pkl'

# Check if the model file exists
if os.path.exists(model_file):
    # Load the model if it exists
    rf_single = joblib.load(model_file)
else:
    # Initialize and train the Random Forest model
    rf_single = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_single.fit(X_train_tfidf, y_train)
    
    # Save the trained model
    joblib.dump(rf_single, model_file)

# Single-label classification function
def classify_text_rf(text):
    text_tfidf = vectorizer.transform([text])
    predicted_class_index = rf_single.predict(text_tfidf)[0]
    predicted_category = le.inverse_transform([predicted_class_index])[0]
    return predicted_category
