import matplotlib.pyplot as plt
import pandas as pd
import psycopg2
from sklearn.metrics import roc_curve, auc
from config import DB_CONFIG
from reporting.report_generator import list_recent_sessions


def plot_auc_from_db(session_id, model='ML'):
    with psycopg2.connect(**DB_CONFIG) as conn:
        df = pd.read_sql("""
                         SELECT true_label, ml_prob, dl_prob
                         FROM test_results
                         WHERE session_id = %s
                         """, conn, params=(str(session_id),))

        df1 = pd.read_sql("""
                         SELECT distinct learning_set
                         FROM test_summary
                         WHERE session_id = %s
                         """, conn, params=(str(session_id),))
    if model.upper() == 'ML':
        y_score = df['ml_prob']
    elif model.upper() == 'DL':
        y_score = df['dl_prob']
    else:
        raise ValueError("Model must be 'ML' or 'DL'")

    # Compute ROC Curve and AUC
    y_true = df['true_label'].map({'ham': 0, 'spam': 1})
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    learn_set_name = df1['learning_set'][0]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'{model} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model} (Session: {session_id})\n Learn dataset: {learn_set_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("Recent testing sessions:")
    sessions = list_recent_sessions(limit=10)

    for s in sessions:
        print(f"  • {s['session_id']} | {s['timestamp']} | {s['learning_set']}")

    chosen = input("\nEnter session_id to generate Excel report: ").strip()
    # model_name = input("\nEnter Model name ('ML' or 'DL') to generate Excel report: ").strip()
    plot_auc_from_db(session_id=chosen, model='ML')
    plot_auc_from_db(session_id=chosen, model='DL')