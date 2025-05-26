## This file is deprecated and should not be used directly.

import os
import glob
import joblib
import tempfile
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from extract_one import extract_one

app = FastAPI(
    title="WiFi Streaming Video Test API",
    description="단일 PCAP 파일을 받아서 표적 스트림을 반환합니다",
    version="1.0.0",
)

MODEL_PATH = "best_svm_model.pkl"
FEATURE_DIR = "single_test_feature"
LABEL_DIR = "single_test_label"

# 미리 필요 디렉토리 생성
os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)


@app.post("/test-pcap/")
async def test_pcap(file: UploadFile = File(...)):
    # 1) 업로드된 PCAP 저장
    if not file.filename.lower().endswith(".pcap"):
        raise HTTPException(400, "pcap 파일만 업로드하세요")
    upload_dir = os.path.join(tempfile.gettempdir(), "pcap_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    local_path = os.path.join(upload_dir, file.filename)
    with open(local_path, "wb") as f:
        f.write(await file.read())

    # 2) 특징 추출
    try:
        extract_one(
            local_path,
            output_dir=FEATURE_DIR,
            labels_dir=LABEL_DIR,
        )
    except Exception as e:
        raise HTTPException(500, f"특징 추출 오류: {e}")

    # 3) CSV → numpy
    csvs = sorted(glob.glob(f"{FEATURE_DIR}/*.csv"))
    if not csvs:
        raise HTTPException(500, "특징 CSV 파일이 생성되지 않았습니다")
    df = pd.read_csv(csvs[0])
    stream_keys = df["stream_key"].tolist()
    X_single = df.drop(columns=["stream_key"]).values

    # 4) 모델 로드 및 예측
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise HTTPException(500, "모델 파일을 찾을 수 없습니다")
    y_pred = model.predict(X_single)

    # 5) 표적 스트림만 필터링
    targets = [s for s, label in zip(stream_keys, y_pred) if label == 1]

    # 6) 결과 반환
    return JSONResponse(
        content={
            # "total_streams": len(stream_keys),
            # "target_count": len(targets),
            "targets": targets,
        }
    )
