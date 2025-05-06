import pandas as pd
import psycopg2
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from config import DB_CONFIG
from reporting.report_generator import list_recent_sessions


def plot_pr_from_db(session_id, model='ML'):
    with psycopg2.connect(**DB_CONFIG) as conn:
        df = pd.read_sql("""
            SELECT true_label, ml_prob, dl_prob
            FROM test_results
            WHERE session_id = %s
        """, conn, params=(str(session_id),))
        df1 = pd.read_sql("""
                          SELECT distinct learning_set, test_set
                          FROM test_summary
                          WHERE session_id = %s
                          """, conn, params=(str(session_id),))
    # Label mapping
    y_true = df['true_label'].map({'ham': 0, 'spam': 1})

    if model.upper() == 'ML':
        y_score = df['ml_prob']
    elif model.upper() == 'DL':
        y_score = df['dl_prob']
    else:
        raise ValueError("Model must be 'ML' or 'DL'")

    valid = y_true.notna() & y_score.notna()
    y_true = y_true[valid]
    y_score = y_score[valid].astype(float)

    if y_true.nunique() < 2:
        print("Cannot compute Precision-Recall: only one class present.")
        return

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    learn_set_name = df1['learning_set'][0]
    test_set_name = df1['test_set'][0]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{model} (AP = {avg_precision:.4f})', color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model} Model\n(Session: {session_id})\n Training dataset: {learn_set_name}. Test dataset: {test_set_name}')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    print("Recent testing sessions:")
    sessions = list_recent_sessions(limit=10)

    for s in sessions:
        print(f"  • {s['session_id']} | {s['timestamp']} | {s['learning_set']} | {s['test_set']}")

    chosen = input("\nEnter session_id to generate Excel report: ").strip()
    # model_name = input("\nEnter Model name ('ML' or 'DL') to generate Excel report: ").strip()
    plot_pr_from_db(session_id=chosen, model='ML')
    plot_pr_from_db(session_id=chosen, model='DL')