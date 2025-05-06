import re

import pandas as pd
from psycopg2.extras import RealDictCursor

from config import *


def list_recent_sessions(limit=5):
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                        SELECT distinct tse.session_id, tse.timestamp, ts.learning_set, ts.test_set
                        FROM test_sessions tse inner join test_summary ts on tse.session_id = ts.session_id
                        ORDER BY tse.timestamp DESC
                        LIMIT %s;
                        """, (limit,))
            return cur.fetchall()



def fetch_session_data(session_id):
    with psycopg2.connect(**DB_CONFIG) as conn:
        summary_df = pd.read_sql("""
                                 SELECT model, accuracy, avg_time, auc, model_size, avg_mem, learning_set, test_set
                                 FROM test_summary
                                 WHERE session_id = %s
                                 """, conn, params=(session_id,))

        details_df = pd.read_sql("""
                                 SELECT text,
                                        true_label,
                                        ml_prediction,
                                        rule_prediction,
                                        hybrid_prediction,
                                        dl_prediction,
                                        ml_time,
                                        rule_time,
                                        hybrid_time,
                                        dl_time,
                                        ml_prob,
                                        dl_prob
                                 FROM test_results
                                 WHERE session_id = %s
                                 """, conn, params=(session_id,))

    return summary_df, details_df


def clean_text_for_excel(s):
    if not isinstance(s, str):
        return s
    return re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', s)


def save_to_excel(session_id, summary_df, details_df, filename=None):
    filename = filename or f"model_test_report_{session_id}.xlsx"
    filename = Path("reports/" + filename)

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        details_df = details_df.map(clean_text_for_excel)
        details_df.to_excel(writer, sheet_name='Details', index=False)

    print(f"Report saved to: {filename.resolve()}")


if __name__ == "__main__":
    while True:
        print("Recent testing sessions:")
        sessions = list_recent_sessions(limit=10)

        for s in sessions:
            print(f"  • {s['session_id']} | {s['timestamp']} | {s['learning_set']} | {s['test_set']}")

        chosen = input("\nEnter session_id to generate Excel report: ").strip()


        try:
            summary_df, details_df = fetch_session_data(chosen)
            save_to_excel(chosen, summary_df, details_df)
        except Exception as e:
            print(f"Failed to generate report: {e}")
