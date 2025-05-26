## This file is used as a testing point for the feature extraction process.

import glob
import os
from sklearn.model_selection import train_test_split

# 1) 파일 경로 수집
g1 = sorted(glob.glob("capture/CAM_OTT/*.pcap"))
g2 = sorted(glob.glob("capture/CAM_LESS_OTT/*.pcap"))
norm = sorted(glob.glob("capture/NO_CAM/*.pcap"))

print("Group1:", len(g1), "files")
print("Group2:", len(g2), "files")
print("Normal:", len(norm), "files")

files = g1 + g2 + norm
labels = ["s1"] * len(g1) + ["s2"] * len(g2) + ["n"] * len(norm)

# 2) 그룹별로 일정 비율(예: 10%)을 테스트셋으로 분리
#    전체 200개 중 20개만 테스트로 뽑으려면 test_size=20/200=0.1
train_files, test_files, train_labels, test_labels = train_test_split(
    files, labels, test_size=20 / 200, stratify=labels, random_state=42
)

# 3) 확인
print(f"Train: {len(train_files)} files, Test: {len(test_files)} files")
# 비율이 맞추어졌는지 그룹별로 한 번 더 체크해보세요
from collections import Counter

print("Train label distribution:", Counter(train_labels))
print("Test  label distribution:", Counter(test_labels))
