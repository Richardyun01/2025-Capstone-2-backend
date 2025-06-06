import os, glob, tempfile
import pandas as pd
from scapy.all import PcapReader, Dot11
from collections import defaultdict
from .rule_engine import stream_features, label_stream

FEATURE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "features", "extracted"
)
os.makedirs(FEATURE_DIR, exist_ok=True)


async def analyze_pcap(file):
    if not file.filename.lower().endswith(".pcap"):
        raise ValueError("PCAP 파일만 업로드 가능합니다.")

    upload_dir = os.path.join(tempfile.gettempdir(), "pcap_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    local_path = os.path.join(upload_dir, file.filename)

    with open(local_path, "wb") as f:
        f.write(await file.read())

    # 기존 CSV 삭제
    for old_file in glob.glob(os.path.join(FEATURE_DIR, "*.csv")):
        os.remove(old_file)

    # Stream 수집 및 분석
    reader = PcapReader(local_path)
    stream_data = defaultdict(
        lambda: {"packet_lengths": [], "packet_intervals": [], "timestamps": []}
    )
    last_time = {}

    for pkt in reader:
        if not pkt.haslayer(Dot11):
            continue
        dot11 = pkt[Dot11]
        if dot11.type != 2 or dot11.addr2 is None:
            continue

        key = dot11.addr2
        plen = len(pkt)
        stream = stream_data[key]
        stream["packet_lengths"].append(plen)
        stream["timestamps"].append(float(pkt.time))

        if key in last_time:
            interval = float(pkt.time) - last_time[key]
            stream["packet_intervals"].append(interval)
        last_time[key] = float(pkt.time)

    reader.close()

    results = []
    for src_mac, stream in stream_data.items():
        features = stream_features(stream)
        suspicious = label_stream(features)
        results.append([src_mac] + features + [suspicious])

    df = pd.DataFrame(
        results,
        columns=[
            "stream_key",
            "total_packets",
            "length_1090_count",
            "max_consecutive_1090",
            "mean_interval",
            "std_interval",
            "mean_length",
            "std_length",
            "inbound_bytes",
            "suspicious",
        ],
    )
    df.to_csv(os.path.join(FEATURE_DIR, "latest.csv"), index=False)

    targets = df[df["suspicious"] == True]["stream_key"].tolist()
    return {"targets": targets}
