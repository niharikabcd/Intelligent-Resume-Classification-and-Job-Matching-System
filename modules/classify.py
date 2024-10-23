from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn import preprocessing

# Load the Hugging Face model and tokenizer
model_name = "ahmedheakl/bert-resume-classification"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load the dataset and prepare the label encoder
dataset_id = 'ahmedheakl/resume-atlas'
from datasets import load_dataset

# Load the dataset
ds = load_dataset(dataset_id, trust_remote_code=True)
label_column = "Category"

# Initialize Label Encoder and fit it to the categories in the dataset
le = preprocessing.LabelEncoder()
le.fit(ds['train'][label_column])

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class_index = torch.argmax(probabilities).item()
    
    # Convert predicted class index to category name
    predicted_category = le.inverse_transform([predicted_class_index])[0]
    return predicted_category

#multiclass-classification
def classify_text_multi(text, threshold=0.95):
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.sigmoid(outputs.logits)
    predicted_classes = (probabilities > threshold).int().tolist()[0]
    job_titles = [le.inverse_transform([idx])[0] for idx, val in enumerate(predicted_classes) if val == 1]
    
    if not job_titles:
        return ["Uncertain Prediction"]
    return job_titles
