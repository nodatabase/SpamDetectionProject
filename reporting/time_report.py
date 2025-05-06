import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", 5432)
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

MODEL_TYPE = 'HYBRID'
OUTPUT_FILE = f'time_matrix_{MODEL_TYPE.lower()}.xlsx'

TRAIN_ORDER = ['Small json', 'UCI', 'Hugging', 'GPT']
TEST_ORDER = ['Small json', 'UCI', 'Hugging']

QUERY = f"""
SELECT
    ts.learning_set AS training_set,
    ts.test_set,
    ts.avg_time
FROM test_summary ts
WHERE ts.model = %s
  AND ts.session_id IN (
      SELECT session_id
      FROM test_sessions
      ORDER BY timestamp DESC
      LIMIT 12
  )
"""

def main():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    try:
        df = pd.read_sql(QUERY, conn, params=(MODEL_TYPE,))

        df['training_set'] = pd.Categorical(df['training_set'], categories=TRAIN_ORDER, ordered=True)
        df['test_set'] = pd.Categorical(df['test_set'], categories=TEST_ORDER, ordered=True)

        pivot_df = df.pivot_table(index='training_set', columns='test_set', values='avg_time')

        pivot_df.to_excel(OUTPUT_FILE, sheet_name='Time Matrix')
        print(f"Saved: {OUTPUT_FILE}")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
