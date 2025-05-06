from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
from joblib import dump
from config import *

def get_training_data(file_path):
    data = pd.read_csv(file_path, encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']
    return data['label'].values, data['message'].values

def get_training_data_hugging(file_path):
    data = pd.read_csv(file_path, sep=',', header=0, names=['text_type', 'text'])
    data.columns = ['text_type', 'text']
    return data['text_type'].values, data['text'].values

def get_training_data_json(file_name):
    data = pd.read_json(file_name)
    data = data[['label', 'message']].dropna(subset=['message'])
    data['message'] = data['message'].astype(str)

    return data['label'].values, data['message'].values


if __name__ == "__main__":

    labels, messages = get_training_data_json(json_file)
    # print(labels)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),
        ('clf', RandomForestClassifier(n_estimators=100))
    ])
    pipeline.fit(messages, labels)

    dump(pipeline, '../model_small_json/ml_spam_model3.joblib')
    print("Model saved as ml_spam_model.joblib")

    new_message = ["Don't miss out on this deal!!! http://scam.link"]
    prediction = pipeline.predict(new_message)
    print("Spam" if prediction[0] == 'spam' else "Not Spam")