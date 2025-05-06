from collections import Counter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from config import *
from models.rule_based import rule_based_spam_score, is_rule_based_spam


def get_training_data(file_path):
    data = pd.read_csv(file_path, encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']
    return data['label'].values, data['message'].values


def weighted_vote(ml_label: str, rule_spam: bool, rule_result_th, ml_weight=0.7, rule_weight=0.3):
    if rule_result_th > 2:
        ml_weight = 0
        rule_weight = 1
    scores = Counter()
    scores['spam'] += ml_weight if ml_label == 'spam' else 0
    scores['spam'] += rule_weight if rule_spam else 0
    scores['ham'] += ml_weight if ml_label == 'ham' else 0
    scores['ham'] += rule_weight if not rule_spam else 0
    return scores.most_common(1)[0][0]


def log_decision(message, true_label, ml_label, rule_spam, final_label):
    print(f"Message: {message}")
    print(f"True Label   : {true_label}")
    print(f"ML Prediction: {ml_label}")
    print(f"Rule-Based   : {'spam' if rule_spam else 'ham'} (score={rule_based_spam_score(message)})")
    print(f"Final Decision: {final_label}")
    print(f"Correct? {'YES' if final_label == true_label else 'NO'}")
    print("-" * 50)

if __name__ == "__main__":
    labels, messages = get_training_data(csv_file_uci)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),
        ('clf', RandomForestClassifier(n_estimators=100))
    ])
    pipeline.fit(messages, labels)

    # Evaluate on a few messages
    num_to_check = 50  # change to 1000+ for real metrics
    correct = 0
    for i in range(num_to_check):
        text = messages[i]
        true_label = labels[i]
        ml_pred = pipeline.predict([text])[0]
        rule_pred = is_rule_based_spam(text)[0]
        final_pred = weighted_vote(ml_pred, rule_pred, 3)

        log_decision(text, true_label, ml_pred, rule_pred, final_pred)

        if final_pred == true_label:
            correct += 1

    print(f"\nFinal Accuracy (voting): {correct}/{num_to_check} = {correct/num_to_check:.2%}")
