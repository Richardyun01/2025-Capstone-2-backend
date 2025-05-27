## This file is used as a testing point for the feature extraction process.

# evaluate.py
import numpy as np
import glob, os, joblib
from sklearn.metrics import classification_report, roc_auc_score

# 1) 캐시된 데이터 로드
X_test = np.load("cache/test/X.npy", allow_pickle=True)
y_test = np.load("cache/test/y.npy", allow_pickle=True)

# 2) 모델 로드
model = joblib.load("best_svm_model.pkl")

# 3) 예측 및 평가
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # 확률 지원 시

print(classification_report(y_test, y_pred, digits=4))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
