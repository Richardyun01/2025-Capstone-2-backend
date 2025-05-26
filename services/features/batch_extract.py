import os
from concurrent.futures import ProcessPoolExecutor

# 폴더 미리 생성
os.makedirs("features/train", exist_ok=True)
os.makedirs("labels/train", exist_ok=True)
os.makedirs("features/test", exist_ok=True)
os.makedirs("labels/test", exist_ok=True)


def batch_extract(file_list, out_feat_dir, out_lbl_dir):
    for p in file_list:
        extract_one(
            p,
            output_dir=os.path.join(out_feat_dir),
            labels_dir=os.path.join(out_lbl_dir),
        )


# 4코어로 동시에 처리
with ProcessPoolExecutor(max_workers=4) as exe:
    exe.submit(batch_extract, train_files, "features/train", "labels/train")
    exe.submit(batch_extract, test_files, "features/test", "labels/test")
