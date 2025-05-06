import pandas as pd
from psycopg2.extras import execute_values
from psycopg2.extras import execute_values

from config import *

psycopg2.extras.register_uuid()



def save_testing_session(session_id, details_df: pd.DataFrame, summary_df: pd.DataFrame):
    conn = get_connection()
    cur = conn.cursor()
    summary_df['AUC'] = pd.to_numeric(summary_df['AUC'], errors='coerce')
    summary_df = summary_df.replace("N/A", pd.NA)
    summary_df = summary_df.replace("NaN", pd.NA)
    summary_df = summary_df.replace("nan", pd.NA)
    summary_df = summary_df.map(lambda x: None if pd.isna(x) else x)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS test_sessions (
            session_id UUID PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS test_results (
            session_id UUID REFERENCES test_sessions(session_id),
            text TEXT,
            true_label TEXT,
            ml_prediction TEXT,
            rule_prediction TEXT,
            hybrid_prediction TEXT,
            dl_prediction TEXT,
            ml_time FLOAT,
            rule_time FLOAT,
            hybrid_time FLOAT,
            dl_time FLOAT,
            ml_prob FLOAT,
            dl_prob FLOAT
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS test_summary (
            session_id UUID REFERENCES test_sessions(session_id),
            model TEXT,
            accuracy FLOAT,
            avg_time FLOAT,
            auc FLOAT,
            model_size FLOAT,
            avg_mem FLOAT,
            learning_set TEXT,
            test_set TEXT
        );
    """)

    cur.execute("INSERT INTO test_sessions (session_id) VALUES (%s);", (session_id,))

    result_rows = [
        (
            session_id,
            row['Text'],
            row['True Label'],
            row['ML Prediction'],
            row['Rule Prediction'],
            row['Hybrid Prediction'],
            row['DL Prediction'],
            row['ML Time'],
            row['Rule Time'],
            row['Hybrid Time'],
            row['DL Time'],
            row['ML Pred Prob'],
            row['DL Pred Prob']
        )
        for _, row in details_df.iterrows()
    ]
    execute_values(cur, """
        INSERT INTO test_results (
            session_id, text, true_label, ml_prediction, rule_prediction,
            hybrid_prediction, dl_prediction, ml_time, rule_time, hybrid_time, dl_time, ml_prob, dl_prob 
        ) VALUES %s;
    """, result_rows)

    summary_rows = [
        (
            session_id,
            row['Model'],
            row['Accuracy (%)'],
            row['Avg Time (s)'],
            row['AUC'],
            row['Model Size (MB)'],
            row['Avg Mem (KiB)'],
            row['Learning set'],
            row['Testing set'],
        )
        for _, row in summary_df.iterrows()
    ]
    execute_values(cur, """
        INSERT INTO test_summary (
            session_id, model, accuracy, avg_time, auc, model_size, avg_mem, learning_set, test_set
        ) VALUES %s;
    """, summary_rows)

    conn.commit()
    cur.close()
    conn.close()

    print(f"Session {session_id} saved to PostgreSQL database.")
