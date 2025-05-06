import os
from pathlib import Path
import psycopg2
from dotenv import load_dotenv


load_dotenv()

csv_file_uci = Path(os.getenv("KAGGLE_PATH")).as_posix()
csv_file_hugging = Path(os.getenv("TELE_PATH")).as_posix()
json_file = Path(os.getenv("WHATSAPP_PATH")).as_posix()
json_gpt_file = Path(os.getenv("GPT_PATH")).as_posix()

telegram_token = os.getenv("TELEGRAM_KEY")

DB_CONFIG = {
    'host': os.getenv("DB_HOST"),
    'port': os.getenv("DB_PORT"),
    'database': os.getenv("DB_NAME"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
}

trained_models = {
    'uci': {
        'ml_model_path': 'model_uci/ml_spam_model.joblib',
        'dl_model_path': 'model_uci/dl_spam_model.keras',
        'tokenizer_path': 'model_uci/dl_tokenizer.pkl',
        'learning_dataset': 'UCI'
    },
    'hugging': {
        'ml_model_path': 'model_hugging/ml_spam_model2.joblib',
        'dl_model_path': 'model_hugging/dl_spam_model2.keras',
        'tokenizer_path': 'model_hugging/dl_tokenizer2.pkl',
        'learning_dataset': 'Hugging'
    },
    'small json': {
        'ml_model_path': 'model_small_json/ml_spam_model3.joblib',
        'dl_model_path': 'model_small_json/dl_spam_model3.keras',
        'tokenizer_path': 'model_small_json/dl_tokenizer3.pkl',
        'learning_dataset': 'Small json'
    }
}

def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )