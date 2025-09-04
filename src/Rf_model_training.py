import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from scipy.stats import randint


# Helper: this fo saving sklearn classification_report

def save_classification_report_png(y_true, y_pred,
                                   title="Random Forest Detailed Classification Report For Cross_dataset_1",
                                   filename="RF_classification_report_Cross_dataset_1.png"):
    rep = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    rows = ['0', '1', 'macro avg', 'weighted avg']
    df = pd.DataFrame(rep).T.loc[rows, ['precision', 'recall', 'f1-score', 'support']]

    # Format
    df['precision'] = df['precision'].map(lambda x: f"{x:.4f}")
    df['recall']    = df['recall'].map(lambda x: f"{x:.4f}")
    df['f1-score']  = df['f1-score'].map(lambda x: f"{x:.4f}")
    df['support']   = df['support'].astype(int).map(lambda x: f"{x:d}")

    fig, ax = plt.subplots(figsize=(7.8, 2.6))
    ax.axis('off')
    table = ax.table(
        cellText=df.values,
        rowLabels=['Class 0', 'Class 1', 'Macro avg', 'Weighted avg'],
        colLabels=['precision', 'recall', 'f1-score', 'support'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    ax.set_title(title, pad=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=250, bbox_inches='tight')
    plt.close()
    print("Saved:", filename)





# Load data

train_data = pd.read_csv('train_cleaned_rf_private.csv')
test_data  = pd.read_csv('test_cleaned_rf_public.csv')

# Define target and feature–target split
target = 'label'
X_train = train_data.drop(columns=[target]); y_train = train_data[target]
X_test  = test_data.drop(columns=[target]);  y_test  = test_data[target]

# Align test features to train schema
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Class distribution
print("\nClass distribution in training set:")
print(y_train.value_counts(normalize=True))


# Model: RandomForest (+ class weights), RandomisedSearchCV

pipeline = Pipeline([('rf', RandomForestClassifier(random_state=50, n_jobs=-1))])

param_distributions = {
    'rf__n_estimators': randint(200, 601),
    'rf__max_depth': [None, 15, 25, 40],
    'rf__min_samples_leaf': randint(1, 5),
    'rf__max_features': ['sqrt', 0.3, 0.5],
    'rf__class_weight': [None, 'balanced'],
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
rand = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=10,
    scoring='average_precision',
    refit=True,
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=1
)
print("\nTuning Random Forest hyperparameters ")
rand.fit(X_train, y_train)
print("\nBest params (by AP):", rand.best_params_)
print(f"Best CV AP: {rand.best_score_:.4f}")
model = rand.best_estimator_


# Prediction
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Metrics Calculation

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
roc_auc   = roc_auc_score(y_test, y_proba)
pr_auc    = average_precision_score(y_test, y_proba)

print("\nModel Performance Metrics on Test Set for Cross_dataset_1:")
print("-------------------------------------------------------------------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")
print(f"PR AUC:    {pr_auc:.4f}")


# Save a table of key metrics as PNG

metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC AUC', 'PR AUC'],
    'Value':  [accuracy,   precision,  recall,   f1,        roc_auc,  pr_auc]
})

fig, ax = plt.subplots(figsize=(6, 2.8))
ax.axis('off')
tbl = ax.table(
    cellText=[[m, f"{v:.4f}"] for m, v in zip(metrics_df['Metric'], metrics_df['Value'])],
    colLabels=['Metric', 'Value'],
    loc='center',
    cellLoc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.4)
ax.set_title('Random Forest Model Performance Metrics on Test Set for Cross_dataset_1 ', pad=12)
plt.tight_layout()
plt.savefig('rf_metrics_table_Cross_dataset_1.png', dpi=200, bbox_inches='tight')
plt.close()


# Detailed classification report per-class

print("\nDetailed Classification Report for Cross_dataset_1:")
print(classification_report(y_test, y_pred, zero_division=0))
save_classification_report_png(
    y_true=y_test,
    y_pred=y_pred,
    title="Random Forest Detailed Classification Report for Cross_dataset_1",
    filename="RF_detailed_classification_report_Cross_dataset_1.png"
)


# Confusion matrix

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("\nConfusion Matrix (counts) for Cross_dataset_1:")
print(f"TN: {tn}  FP: {fp}  FN: {fn}  TP: {tp}")

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Normal (0)', 'Attack (1)'],
            yticklabels=['Normal (0)', 'Attack (1)'])
plt.title('Random Forest Confusion Matrix for Cross_dataset_1'
          '')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('rf_confusion_matrix_Cross_dataset_1.png', dpi=200)
plt.close()


# ROC Curve

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Random Forest ROC Curve for Cross_dataset_1')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('rf_roc_curve_Cross_dataset_1.png', dpi=200)
plt.close()


# Precision–Recall Curve

prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(rec_curve, prec_curve, label=f'AP (PR AUC) = {pr_auc:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Random Forest Precision–Recall Curve for Cross_dataset_1')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('rf_precision_recall_curve_Cross_dataset_1.png', dpi=200)
plt.close()
