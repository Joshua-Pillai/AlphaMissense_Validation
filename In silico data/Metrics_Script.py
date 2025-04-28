# script was run in Google Colab with pre-installed libraries.

import pandas as pd

df = pd.read_csv('/content/Test Run327 - Sheet1.csv')
df['AM_Pathogenic'] = df['Study Classification'].apply(lambda x: 1 if x == 'Pathogenic' else 0)
df['AM_Benign'] = df['Study Classification'].apply(lambda x: 1 if x == 'Benign' else 0)
df['Pred_Model1'] = df['AM'].apply(lambda x: 1 if x > 0.56 else 0)
df['Pred_Model2'] = df['AM'].apply(lambda x: 1 if x < 0.34 else 0)

print(df[['Study Classification', 'AM', 'Pred_Model1', 'Pred_Model2']].head())

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def AM_performance(y_true, y_pred, model_name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred)

AM_performance(df['True_Model1'], df['Pred_Model1'], "Model 1: Pathogenic vs All")
AM_performance(df['True_Model2'], df['Pred_Model2'], "Model 2: Benign vs All")
