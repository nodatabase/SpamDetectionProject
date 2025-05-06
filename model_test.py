import pickle
import time
import uuid
import pandas as pd
import psutil
import psycopg2.extras
import tensorflow as tf
from joblib import load
from sklearn.metrics import roc_auc_score
# noinspection PyUnresolvedReferences
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import *
from reporting.db_saver import save_testing_session
from models.hybrid import weighted_vote
from models.rule_based import is_rule_based_spam

psycopg2.extras.register_uuid()
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def measure_memory_block(code_block): # needs improvement
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    result = code_block()
    mem_after = process.memory_info().rss
    mem_used_kb = (mem_after - mem_before) / 1024  # in KB
    return result, round(mem_used_kb, 3)


def get_model_size(path):
    if os.path.exists(path):
        return round(os.path.getsize(path) / (1024 * 1024), 4)
    return 0.0

def compute_auc_res(y_true, y_scores):
    try:
        return round(roc_auc_score(y_true, y_scores), 4)
    except:
        return "N/A"


def load_test_data(file_path):
    ext = os.path.splitext(file_path)[-1].lower()

    encodings_to_try = ['utf-8', 'ISO-8859-1', 'windows-1252']

    for encoding in encodings_to_try:
        try:
            if ext == '.json':
                df = pd.read_json(file_path)
            elif ext in ['.csv', '.txt']:
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    usecols=[0, 1],
                    names=['label', 'message'],
                    skiprows=1
                )
            else:
                raise ValueError("Unsupported file type.")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("Unable to decode file using tried encodings.")

    if 'label' not in df.columns or 'message' not in df.columns:
        raise ValueError("Input file must contain 'label' and 'message' columns.")

    df = df[['label', 'message']].dropna(subset=['message'])
    df['message'] = df['message'].astype(str)
    return df


def evaluate_models(test_df, model_name, test_dataset):
    ml_model_path = trained_models[model_name]['ml_model_path']
    dl_model_path = trained_models[model_name]['dl_model_path']
    tokenizer_path = trained_models[model_name]['tokenizer_path']
    testing_dataset = trained_models[model_name]['learning_dataset']
    ml_model = load(ml_model_path)
    dl_model = tf.keras.models.load_model(dl_model_path)
    with open(tokenizer_path, 'rb') as f:
        dl_tokenizer = pickle.load(f)

    max_length = 50

    stats = {
        'ml': {'correct': 0, 'time': 0},
        'rule': {'correct': 0, 'time': 0},
        'hybrid': {'correct': 0, 'time': 0},
        'dl': {'correct': 0, 'time': 0},
        'total': len(test_df)
    }

    records = []
    ml_probs, dl_probs = [], []
    true_labels = []
    ml_mem_logs, dl_mem_logs, rule_mem_logs, hybrid_mem_logs = [], [], [], []

    for _, row in test_df.iterrows():
        text = row['message']
        label = row['label']
        true_labels.append(1 if label == 'spam' else 0)

        # ML
        start = time.perf_counter()
        def ml_call(): return ml_model.predict([text])[0]
        ml_pred, ml_mem = measure_memory_block(ml_call)
        ml_prob = ml_model.predict_proba([text])[0][1]
        ml_time = time.perf_counter() - start
        stats['ml']['time'] += ml_time
        if ml_pred == label:
            stats['ml']['correct'] += 1
        ml_probs.append(ml_prob)
        ml_mem_logs.append(ml_mem)

        # Rule
        rule_result_th = is_rule_based_spam(text)[1]
        start = time.perf_counter()
        def rule_call(): return 'spam' if is_rule_based_spam(text)[0] else 'ham'
        rule_pred, rule_mem = measure_memory_block(rule_call)
        rule_time = time.perf_counter() - start
        stats['rule']['time'] += rule_time
        if rule_pred == label:
            stats['rule']['correct'] += 1
        rule_mem_logs.append(rule_mem)


        # Hybrid
        start = time.perf_counter()
        def hybrid_call(): return weighted_vote(ml_pred, rule_pred == 'spam', rule_result_th)
        hybrid_pred, hybrid_mem = measure_memory_block(hybrid_call)
        hybrid_time = time.perf_counter() - start
        if rule_result_th > 2:
            hybrid_mem = hybrid_mem + rule_mem
            hybrid_time = hybrid_time + rule_time
        else:
            hybrid_mem = hybrid_mem + ml_mem + rule_mem # add previously computed result of ml and rule
            hybrid_time = hybrid_time + ml_time + rule_time # add previously computed result
        stats['hybrid']['time'] += hybrid_time
        if hybrid_pred == label:
            stats['hybrid']['correct'] += 1
        hybrid_mem_logs.append(hybrid_mem)


        # DL
        seq = dl_tokenizer.texts_to_sequences([text])
        # print(seq, text)
        padded = pad_sequences(seq, maxlen=max_length, padding='post')
        start = time.perf_counter()
        def dl_call(): return dl_model.predict(padded, verbose=0)[0][0]
        dl_prob, dl_mem = measure_memory_block(dl_call)
        dl_pred = 'spam' if dl_prob > 0.5 else 'ham'
        dl_time = time.perf_counter() - start
        stats['dl']['time'] += dl_time
        if dl_pred == label:
            stats['dl']['correct'] += 1
        dl_probs.append(dl_prob)
        dl_mem_logs.append(dl_mem)
        # print(dl_prob)
        records.append({
            'Text': text,
            'True Label': label,
            'ML Prediction': ml_pred,
            'Rule Prediction': rule_pred,
            'Hybrid Prediction': hybrid_pred,
            'DL Prediction': dl_pred,
            'ML Time': ml_time,
            'Rule Time': rule_time,
            'Hybrid Time': hybrid_time,
            'DL Time': dl_time,
            'ML Pred Prob': ml_prob,
            'DL Pred Prob': dl_prob
        })

    # Summary
    summary_data = []
    ml_auc = compute_auc_res(true_labels, ml_probs)
    dl_auc = compute_auc_res(true_labels, dl_probs)

    for model in ['ml', 'rule', 'hybrid', 'dl']:
        acc = stats[model]['correct'] / stats['total']
        avg_time = stats[model]['time'] / stats['total']
        if model == 'ml':
            model_size = get_model_size(ml_model_path)
            avg_mem = round(sum(ml_mem_logs) / len(ml_mem_logs), 4)
            auc = ml_auc
        elif model == 'dl':
            model_size = get_model_size(dl_model_path)
            avg_mem = round(sum(dl_mem_logs) / len(dl_mem_logs), 4)
            auc = dl_auc
        elif model == 'rule':
            model_size = 'N/A'
            avg_mem = round(sum(rule_mem_logs) / len(rule_mem_logs), 4)
            auc = 'N/A'
        else:  # hybrid
            model_size = 'N/A'
            avg_mem = round(sum(hybrid_mem_logs) / len(hybrid_mem_logs), 4)
            auc = 'N/A'

        summary_data.append({
            'Model': model.upper(),
            'Accuracy (%)': round(acc * 100, 5),
            'Avg Time (s)': round(avg_time, 6),
            'AUC': auc,
            'Model Size (MB)': model_size,
            'Avg Mem (KiB)': avg_mem,
            'Learning set': testing_dataset,
            'Testing set': test_dataset
        })

    summary_df = pd.DataFrame(summary_data)
    details_df = pd.DataFrame(records)
    print(summary_df)
    print(details_df)

    session_id = uuid.uuid4()
    save_testing_session(session_id, details_df, summary_df)

if __name__ == "__main__":
    df_test = load_test_data(json_file)
    evaluate_models(df_test, 'uci', 'Small json')
    evaluate_models(df_test, 'hugging', 'Small json')
    evaluate_models(df_test, 'small json', 'Small json')

    df_test = load_test_data(csv_file_uci)
    evaluate_models(df_test, 'uci', 'UCI')
    evaluate_models(df_test, 'hugging', 'UCI')
    evaluate_models(df_test, 'small json', 'UCI')


    df_test = load_test_data(csv_file_hugging)
    evaluate_models(df_test, 'uci', 'Hugging')
    evaluate_models(df_test, 'hugging', 'Hugging')
    evaluate_models(df_test, 'small json', 'Hugging')


    df_test = load_test_data(json_gpt_file)
    evaluate_models(df_test, 'uci', 'GPT')
    evaluate_models(df_test, 'hugging', 'GPT')
    evaluate_models(df_test, 'small json', 'GPT')
