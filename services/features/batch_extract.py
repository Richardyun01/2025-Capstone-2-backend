import os
import glob
from concurrent.futures import ProcessPoolExecutor
from extract_one import extract_one

# PCAP 파일 리스트
train_files = glob.glob("capture/train/*.pcap")
test_files = glob.glob("capture/test/*.pcap")

# 출력 폴더 생성 (import 시가 아니라, main 안에서 해도 무방)
FEATURE_DIRS = [
    ("features/train", "labels/train"),
    ("features/test", "labels/test"),
]


def batch_extract(file_list, out_feat_dir, out_lbl_dir):
    os.makedirs(out_feat_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    for p in file_list:
        extract_one(
            p,
            output_dir=out_feat_dir,
            labels_dir=out_lbl_dir,
        )


if __name__ == "__main__":
    # Windows spawn 안전 처리
    import multiprocessing

    multiprocessing.freeze_support()

    # 필요한 폴더들 미리 생성
    for feat_dir, lbl_dir in FEATURE_DIRS:
        os.makedirs(feat_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

    # 병렬 실행
    with ProcessPoolExecutor(max_workers=4) as exe:
        exe.submit(batch_extract, train_files, *FEATURE_DIRS[0])
        exe.submit(batch_extract, test_files, *FEATURE_DIRS[1])
