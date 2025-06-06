import argparse
import csv
import numpy as np
from collections import defaultdict
from scapy.all import PcapReader, Dot11


def max_consecutive_packets(lengths, target=1090, tol=50):
    max_count = count = 0
    for l in lengths:
        if abs(l - target) <= tol:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
    return max_count


def compute_length_entropy(lengths, bins=10):
    """
    lengths: 1D 리스트 또는 ndarray of packet lengths
    bins: 히스토그램 빈 개수
    결과: Shannon entropy (float)
    """
    arr = np.array(lengths, dtype=float)
    if arr.size == 0:
        return 0.0
    counts, _ = np.histogram(arr, bins=bins, density=False)
    probs = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def stream_features(stream):
    """
    stream: {
      "packet_lengths": [...],
      "packet_intervals": [...],
      "timestamps": [...]
    }
    위 3개 리스트를 받아 아래의 9개 특징을 계산하여 반환.
    반환 순서:
      0: total_packets
      1: length_1090_count
      2: max_consecutive_1090
      3: mean_interval
      4: std_interval
      5: mean_length
      6: std_length
      7: inbound_bytes
      8: length_entropy
    """
    lengths = np.array(stream["packet_lengths"], dtype=float)
    intervals = np.array(stream["packet_intervals"], dtype=float)

    total_packets = len(lengths)
    inbound_bytes = float(np.sum(lengths))
    length_1090_count = int(np.sum(np.abs(lengths - 1090) <= 50))

    # 기존 통계
    mean_interval = float(np.mean(intervals)) if intervals.size > 0 else 0.0
    std_interval = float(np.std(intervals)) if intervals.size > 0 else 0.0
    mean_length = float(np.mean(lengths)) if lengths.size > 0 else 0.0
    std_length = float(np.std(lengths)) if lengths.size > 0 else 0.0
    max_consec_1090 = int(max_consecutive_packets(lengths.tolist()))

    # 추가: 패킷 크기 분포 엔트로피
    length_entropy = compute_length_entropy(lengths, bins=10)

    features = [
        total_packets,  # 0
        length_1090_count,  # 1
        max_consec_1090,  # 2
        mean_interval,  # 3
        std_interval,  # 4
        mean_length,  # 5
        std_length,  # 6
        inbound_bytes,  # 7
        length_entropy,  # 8
    ]
    return features


def label_stream(features):
    """
    features 배열 인덱스:
      0: total_packets
      1: length_1090_count
      2: max_consecutive_1090
      3: mean_interval
      4: std_interval      (unused)
      5: mean_length       (unused)
      6: std_length        (unused)
      7: inbound_bytes     (unused)
      8: length_entropy
    공통 A·B 탐지 조건:
      1) total_packets >= 100 OR total_packets == 1
      2) mean_interval <= 0.5
      3) length_entropy < 1.0
    """
    total_packets = features[0]
    mean_interval = features[3]
    length_entropy = features[8]

    cond_count = (total_packets >= 100) or (total_packets == 1)
    cond_interval = mean_interval <= 0.6
    cond_size = length_entropy < 0.9

    return int(cond_count and cond_interval and cond_size)


def cmd_extract(args):
    reader = PcapReader(args.pcap)
    stream_data = defaultdict(
        lambda: {"packet_lengths": [], "packet_intervals": [], "timestamps": []}
    )
    last_time = {}

    TIMEOUT = 5.0  # 초과하는 간격은 '간헐적 burst'로 보고 interval 계산에서 제외

    for pkt in reader:
        if not pkt.haslayer(Dot11):
            continue
        dot11 = pkt[Dot11]
        if dot11.type != 2 or dot11.addr2 is None or dot11.addr1 is None:
            continue

        src_mac = dot11.addr2
        dest_mac = dot11.addr1
        key = (src_mac, dest_mac)

        plen = len(pkt)
        t = float(pkt.time)

        stream = stream_data[key]
        stream["packet_lengths"].append(plen)
        stream["timestamps"].append(t)

        if key in last_time:
            dt = t - last_time[key]
            if dt <= TIMEOUT:
                stream["packet_intervals"].append(dt)
        last_time[key] = t

    reader.close()

    # CSV 헤더: src_mac, dest_mac, 그리고 9개 feature 순서대로
    header = [
        "src_mac",
        "dest_mac",
        "total_packets",
        "length_1090_count",
        "max_consecutive_1090",
        "mean_interval",
        "std_interval",
        "mean_length",
        "std_length",
        "inbound_bytes",
        "length_entropy",
    ]

    with open(args.output, "w", newline="") as wf, open(args.labels, "w") as lf:
        writer = csv.writer(wf)
        writer.writerow(header)

        for (src_mac, dest_mac), stream in stream_data.items():
            if len(stream["packet_lengths"]) < 1:
                continue  # 스트림이 비어 있으면 제외

            features = stream_features(stream)
            writer.writerow([src_mac, dest_mac] + features)
            lf.write(f"{label_stream(features)}\n")

    print(f"Extracted {len(stream_data)} unique (src, dest) streams")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Wi-Fi stream features by (src, dest) MAC from a single PCAP"
    )
    parser.add_argument("-f", "--pcap", required=True, help="PCAP 파일 경로")
    parser.add_argument("-o", "--output", required=True, help="결과 CSV 파일 경로")
    parser.add_argument("-l", "--labels", required=True, help="라벨 파일(.lbl) 경로")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cmd_extract(args)
