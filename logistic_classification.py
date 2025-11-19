
import os
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set()

# -----------------------------
# CONFIG
# -----------------------------
# Default dataset path (change if needed)
DEFAULT_PATH = "C:\\Users\\chand\\OneDrive\\Documents\\DATASET\\data.csv"  # uploaded file location
OUT_DIR = Path('lr_outputs')
OUT_DIR.mkdir(exist_ok=True)

# Optional: set a specific target column name here if auto-detection fails:
FORCE_TARGET = None  # e.g. 'target' or 'Diagnosis' ; set to None to auto-detect

# -----------------------------
# Helpers
# -----------------------------
def load_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Update DEFAULT_PATH.")
    if path.suffix.lower() in ('.xls', '.xlsx'):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return df

def detect_target(df):
    # 1) explicit candidate names
    common = ['target', 'label', 'y', 'class', 'outcome', 'survived', 'diagnosis', 'result']
    for name in df.columns:
        if name.lower() in common:
            return name

    # 2) look for binary columns with exactly 2 unique non-null values (excluding index-like cols)
    for col in df.columns[::-1]:  # prefer right-most columns last -> often target
        if df[col].nunique(dropna=True) == 2:
            return col

    # 3) fallback: last column
    return df.columns[-1]

def binarize_target(series):
    # Map to 0/1
    vals = series.dropna().unique().tolist()
    if len(vals) != 2:
        raise ValueError("Target does not look binary.")
    # if already 0/1 or bool
    if set(map(str, vals)) <= {'0','1','0.0','1.0','False','True'}:
        return series.astype(int)
    # choose mapping: map the first encountered to 0, second to 1
    mapping = {vals[0]: 0, vals[1]: 1}
    return series.map(mapping).astype(int)

def evaluate_and_report(y_true, y_pred, y_prob, out_prefix='report'):
    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)

    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")
    print("\nClassification report:\n", classification_report(y_true, y_pred))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_path = OUT_DIR / f'{out_prefix}_confusion_matrix.csv'
    pd.DataFrame(cm, index=['actual_0','actual_1'], columns=['pred_0','pred_1']).to_csv(cm_path)
    print(f"Saved confusion matrix CSV to {cm_path}")

    # ROC curve & save
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_path = OUT_DIR / f'{out_prefix}_roc.png'
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC curve to {roc_path}")

    # return metrics dict
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=roc_auc,
                roc_fpr=fpr.tolist(), roc_tpr=tpr.tolist(), roc_thresholds=thresholds.tolist())

# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading dataset:", DEFAULT_PATH)
    df = load_data(DEFAULT_PATH)
    print("Data shape:", df.shape)
    print("Columns:", list(df.columns))

    # Detect target
    if FORCE_TARGET:
        target_col = FORCE_TARGET
    else:
        target_col = detect_target(df)
    print("Detected target column:", target_col)

    # Move target out & basic checks
    y_raw = df[target_col]
    X = df.drop(columns=[target_col]).copy()
    print("Features shape:", X.shape)

    # Binarize target if needed
    try:
        y = binarize_target(y_raw)
    except Exception as e:
        # If target already numeric with only 0/1, cast
        if pd.api.types.is_numeric_dtype(y_raw):
            uniq = sorted(y_raw.dropna().unique().tolist())
            if set(uniq) <= {0,1}:
                y = y_raw.astype(int)
            else:
                raise RuntimeError("Target column is numeric but not binary. Set FORCE_TARGET manually.") from e
        else:
            raise

    # Simple report: value counts
    print("Target value counts:")
    print(y.value_counts())

    # Detect feature types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # If a categorical column is numeric-coded but low cardinality, treat as categorical
    for col in numeric_cols[:]:  # copy
        if X[col].nunique() <= 10 and X[col].dtype.kind in 'iu':
            # Heuristic: if integer and small unique count, probably categorical; keep as numeric otherwise.
            # Keep it numeric unless user wants otherwise; here we leave it numeric.
            pass

    print("Numeric cols:", numeric_cols)
    print("Categorical cols:", categorical_cols)

    # Build column transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ]) if categorical_cols else None

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
        ] + ([('cat', categorical_transformer, categorical_cols)] if categorical_cols else []),
        remainder='drop'
    )

    # Train/test split (stratify on y so class balance preserved)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Train/test sizes:", X_train.shape, X_test.shape)

    # Build pipeline with logistic regression
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(solver='liblinear', max_iter=1000))
    ])

    print("\nFitting LogisticRegression pipeline...")
    clf.fit(X_train, y_train)

    # Predict & probabilities
    y_pred = clf.predict(X_test)
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)[:, 1]
    else:
        # fallback to decision_function -> sigmoid
        dec = clf.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-dec))

    # Evaluate & save report
    metrics = evaluate_and_report(y_test, y_pred, y_prob, out_prefix='logreg')

    # Save model
    model_path = OUT_DIR / 'logistic_pipeline.joblib'
    joblib.dump(clf, model_path)
    print(f"Saved trained pipeline to {model_path}")

    # Save test predictions with probabilities
    results_df = X_test.copy().reset_index(drop=True)
    results_df['y_true'] = y_test.reset_index(drop=True)
    results_df['y_pred'] = y_pred
    results_df['y_prob'] = y_prob
    results_csv = OUT_DIR / 'test_predictions.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"Saved test predictions to {results_csv}")

    # Threshold tuning: show precision/recall at a few thresholds and plot precision/recall vs threshold
    thresholds = np.linspace(0.0, 1.0, 101)
    pr_data = []
    for t in thresholds:
        preds_t = (y_prob >= t).astype(int)
        pr = precision_score(y_test, preds_t, zero_division=0)
        rc = recall_score(y_test, preds_t, zero_division=0)
        f1 = f1_score(y_test, preds_t, zero_division=0)
        pr_data.append((t, pr, rc, f1))
    pr_df = pd.DataFrame(pr_data, columns=['threshold','precision','recall','f1'])
    pr_df.to_csv(OUT_DIR / 'threshold_precision_recall.csv', index=False)
    print(f"Saved threshold precision/recall to {OUT_DIR/'threshold_precision_recall.csv'}")

    # Plot precision & recall vs threshold
    plt.figure(figsize=(7,4))
    plt.plot(pr_df['threshold'], pr_df['precision'], label='precision')
    plt.plot(pr_df['threshold'], pr_df['recall'], label='recall')
    plt.plot(pr_df['threshold'], pr_df['f1'], label='f1')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision / Recall / F1 vs Threshold')
    plt.legend()
    plt.tight_layout()
    thr_plot = OUT_DIR / 'precision_recall_vs_threshold.png'
    plt.savefig(thr_plot)
    plt.close()
    print(f"Saved threshold plot to {thr_plot}")

    # Plot probability (sigmoid-like) distribution for each class
    plt.figure(figsize=(7,4))
    sns.histplot(y_prob[y_test==0], label='class 0', stat='density', element='step', fill=False)
    sns.histplot(y_prob[y_test==1], label='class 1', stat='density', element='step', fill=False)
    plt.xlabel('Predicted probability (class 1)')
    plt.title('Predicted probability distributions by true class')
    plt.legend()
    prob_dist_plot = OUT_DIR / 'probability_dist_by_class.png'
    plt.tight_layout()
    plt.savefig(prob_dist_plot)
    plt.close()
    print(f"Saved probability distribution plot to {prob_dist_plot}")

    # Save metrics JSON-like csv
    metrics_df = pd.DataFrame([{
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'roc_auc': metrics['roc_auc']
    }])
    metrics_df.to_csv(OUT_DIR / 'evaluation_metrics.csv', index=False)
    print(f"Saved evaluation metrics to {OUT_DIR/'evaluation_metrics.csv'}")

    print("\nAll outputs saved to:", OUT_DIR)
    print("Done.")

if __name__ == '__main__':
    main()
