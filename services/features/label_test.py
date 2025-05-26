## This file is used as a testing point for the feature extraction process.

import glob
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1) 비교할 MAC 프리픽스 리스트
TARGET_PREFIXES = ["02:00:11:22"]  # 앞 6자리(콜론 제거 혹은 포함)


def auto_label_stream(stream_key):
    """stream_key = 'src|dst' 형태.
    src 또는 dst가 TARGET_PREFIXES 중 하나로 시작하면 1, 아니면 0."""
    src, dst = stream_key.split("|")
    for pref in TARGET_PREFIXES:
        if src.lower().replace(":", "").startswith(
            pref.replace(":", "")
        ) or dst.lower().replace(":", "").startswith(pref.replace(":", "")):
            return 1
    return 0


def evaluate_one(feat_csv, lbl_file):
    # feature CSV 로딩 (header 포함)
    df = pd.read_csv(feat_csv)
    # manual label 로딩 (한 줄에 0/1)
    manual = pd.read_csv(lbl_file, header=None, names=["manual"])
    # 자동 라벨 생성
    df["auto"] = df["stream_key"].apply(auto_label_stream)
    df["manual"] = manual["manual"]

    y_true = df["manual"]
    y_pred = df["auto"]

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    return acc, cm, report


def main():
    feats = sorted(glob.glob("features/*.csv"))
    lbls = sorted(glob.glob("labels/*.lbl"))
    assert len(feats) == len(lbls), "features 와 labels 파일 개수 불일치"

    total_acc = []
    for f, l in zip(feats, lbls):
        acc, cm, report = evaluate_one(f, l)
        print(f"\n=== {os.path.basename(f)} / {os.path.basename(l)} ===")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(report)
        total_acc.append(acc)

    print(f"\n전체 파일 평균 정확도: {sum(total_acc)/len(total_acc):.4f}")


if __name__ == "__main__":
    main()
