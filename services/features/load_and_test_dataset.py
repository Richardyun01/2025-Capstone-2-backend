## This file is deprecated and should not be used directly.

import os
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV

# 캐시 디렉터리 및 파일 경로
CACHE_DIR = "cache"
TRAIN_CACHE = os.path.join(CACHE_DIR, "train")
TEST_CACHE = os.path.join(CACHE_DIR, "test")
TRAIN_XNPY = os.path.join(TRAIN_CACHE, "X.npy")
TRAIN_YNPY = os.path.join(TRAIN_CACHE, "y.npy")
TEST_XNPY = os.path.join(TEST_CACHE, "X.npy")
TEST_YNPY = os.path.join(TEST_CACHE, "y.npy")


def build_and_cache(feat_glob, lbl_glob, cache_dir):
    """CSV 파일들로부터 X, y를 만들고 .npy로 캐시."""
    feats = sorted(glob.glob(feat_glob))
    lbls = sorted(glob.glob(lbl_glob))
    X_list = []
    for f in feats:
        df = pd.read_csv(f)
        X_list.append(df.drop(columns=["stream_key"]).values)
    X = np.vstack(X_list)
    y = np.hstack([np.loadtxt(l, dtype=int) for l in lbls])
    # 디렉터리 생성
    os.makedirs(cache_dir, exist_ok=True)
    # 저장
    np.save(os.path.join(cache_dir, "X.npy"), X)
    np.save(os.path.join(cache_dir, "y.npy"), y)
    print(f"Cached {cache_dir}: X shape {X.shape}, y shape {y.shape}")
    return X, y


# 1. 캐시된 파일이 있으면 로드, 없으면 빌드 & 캐시
if os.path.exists(TRAIN_XNPY) and os.path.exists(TRAIN_YNPY):
    # allow_pickle=True 추가
    X_train = np.load(TRAIN_XNPY, allow_pickle=True)
    y_train = np.load(TRAIN_YNPY, allow_pickle=True)
    print(f"Loaded cached train: X shape {X_train.shape}, y shape {y_train.shape}")
else:
    X_train, y_train = build_and_cache(
        "features/train/*.csv", "labels/train/*.lbl", TRAIN_CACHE
    )

if os.path.exists(TEST_XNPY) and os.path.exists(TEST_YNPY):
    X_test = np.load(TEST_XNPY, allow_pickle=True)
    y_test = np.load(TEST_YNPY, allow_pickle=True)
    print(f"Loaded cached test:  X shape {X_test.shape},  y shape {y_test.shape}")
else:
    X_test, y_test = build_and_cache(
        "features/test/*.csv", "labels/test/*.lbl", TEST_CACHE
    )

print("Train shape:", X_train.shape, y_train.shape)
print("Test  shape:", X_test.shape, y_test.shape)

# 2. Pipeline + Cross-Validation
pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", class_weight="balanced", probability=True)),
    ]
)
scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
print("5-fold CV ROC AUC scores:", scores)
print("Mean CV ROC AUC:", scores.mean())

# 3. 최종 모델 학습 및 테스트
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

print("\n=== Test Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))
print("Test ROC AUC:", roc_auc_score(y_test, y_prob))

# 4. Grid Search: 최적의 하이퍼파라미터 찾기
param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.01, 0.1, 1],
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Best CV AUC:", grid.best_score_)

# 5. 최적 모델 저장
best_model = grid.best_estimator_
joblib.dump(best_model, "best_svm_model.pkl")
print("Saved best SVM model to best_svm_model.pkl")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traineval SVM or test a single PCAP stream"
    )
    parser.add_argument(
        "--test-pcap",
        help="단일 PCAP 파일 경로(.pcap)를 지정하면, 추출→예측→표적 스트림만 출력",
    )
    args = parser.parse_args()

    if args.test_pcap:
        from batch_extract import extract_one

        # 1) 특징 추출
        single_feat_dir = "single_test_feature"
        single_lbl_dir = "single_test_label"
        os.makedirs(single_feat_dir, exist_ok=True)
        os.makedirs(single_lbl_dir, exist_ok=True)
        extract_one(
            args.test_pcap,
            output_dir=single_feat_dir,
            labels_dir=single_lbl_dir,
        )

        # 2) CSV→numpy 로딩
        import glob
        import pandas as pd

        csvs = sorted(glob.glob(f"{single_feat_dir}/*.csv"))
        if not csvs:
            raise FileNotFoundError("특징 CSV가 없습니다.")
        df = pd.read_csv(csvs[0])
        stream_keys = df["stream_key"].values
        X_single = df.drop(columns=["stream_key"]).values

        # 3) 예측 & 출력
        model = joblib.load("best_svm_model.pkl")
        y_pred = model.predict(X_single)
        targets = stream_keys[y_pred == 1]

        print("\n=== 단일 PCAP 테스트 결과 ===")
        if len(targets):
            print("표적 스트림:")
            for s in targets:
                print(" ", s)
        else:
            print("표적으로 분류된 스트림이 없습니다.")
        exit(0)
