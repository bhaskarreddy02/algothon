import pandas as pd
import numpy as np

df = pd.read_csv("predictions/holdout_predictions.csv")
p = df["failure_probability"]
y = df["Machine failure"]

print("=== HOLDOUT PROBABILITY DISTRIBUTION ===")
print("Total holdout samples:", len(df))
print("Samples with p < 0.05:", (p < 0.05).sum())
print("Samples with 0.05 <= p < 0.35:", ((p >= 0.05) & (p < 0.35)).sum())
print("Samples with 0.35 <= p < 0.4951:", ((p >= 0.35) & (p < 0.4951)).sum())
print("Samples with 0.4951 <= p < 0.5000:", ((p >= 0.4951) & (p < 0.5000)).sum())
print("Samples with 0.5000 <= p < 0.8415:", ((p >= 0.5000) & (p < 0.8415)).sum())
print("Samples with p >= 0.8415:", (p >= 0.8415).sum())

print("\nSamples between 0.4951 and 0.8415:")
subset = df[(p >= 0.4951) & (p < 0.8415)]
print(subset[["UDI", "Machine failure", "failure_probability", "Type", "Tool wear [min]", "Torque [Nm]"]])

print("\nConfusion Matrix at tau = 0.4951 (High Recall):")
pred_hr = (p >= 0.4951).astype(int)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
print(confusion_matrix(y, pred_hr))
print(f"Prec: {precision_score(y, pred_hr):.4f}, Rec: {recall_score(y, pred_hr):.4f}, F1: {f1_score(y, pred_hr):.4f}")

print("\nConfusion Matrix at tau = 0.5000 (Default):")
pred_def = (p >= 0.5000).astype(int)
print(confusion_matrix(y, pred_def))
print(f"Prec: {precision_score(y, pred_def):.4f}, Rec: {recall_score(y, pred_def):.4f}, F1: {f1_score(y, pred_def):.4f}")

print("\nConfusion Matrix at tau = 0.8415 (Max F1):")
pred_f1 = (p >= 0.8415).astype(int)
print(confusion_matrix(y, pred_f1))
print(f"Prec: {precision_score(y, pred_f1):.4f}, Rec: {recall_score(y, pred_f1):.4f}, F1: {f1_score(y, pred_f1):.4f}")
