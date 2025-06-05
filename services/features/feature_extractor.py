import os
import glob
import joblib
import tempfile
import pandas as pd
from .extract_one import extract_one

# 현재 파일 기준 경로 지정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_svm_model.pkl")
FEATURE_DIR = os.path.join(BASE_DIR, "single_test_feature")
LABEL_DIR = os.path.join(BASE_DIR, "single_test_label")

# 특징, 라벨 디렉토리 생성
os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)


async def analyze_pcap(file):
    # 업로드된 PCAP 저장
    if not file.filename.lower().endswith(".pcap"):
        raise ValueError("PCAP 파일만 업로드 가능합니다.")

    upload_dir = os.path.join(tempfile.gettempdir(), "pcap_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    local_path = os.path.join(upload_dir, file.filename)

    with open(local_path, "wb") as f:
        f.write(await file.read())

    # 특징 디렉토리 초기화 (기존 CSV 삭제)
    for old_file in glob.glob(os.path.join(FEATURE_DIR, "*.csv")):
        os.remove(old_file)

    # 특징 추출
    try:
        extract_one(
            local_path,
            output_dir=FEATURE_DIR,
            labels_dir=LABEL_DIR,
        )
    except Exception as e:
        raise RuntimeError(f"특징 추출 오류: {e}")

    # 새로 만든 특징만 읽기
    csvs = sorted(glob.glob(os.path.join(FEATURE_DIR, "*.csv")))
    if not csvs:
        raise FileNotFoundError("특징 CSV 파일이 생성되지 않았습니다.")

    df = pd.read_csv(csvs[-1])  # 최신 파일만 처리
    src_macs = df["src_mac"].tolist()
    X_single = df.drop(columns=["src_mac", "dest_mac"]).values

    # 모델 로드 및 예측
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("모델 파일을 찾을 수 없습니다.")

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_single)

    # 표적 스트림 필터링
    targets = [s for s, label in zip(src_macs, y_pred) if label == 1]

    return {
        "targets": targets,
    }
