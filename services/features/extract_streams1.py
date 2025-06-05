## src/dest 쌍 기반 스트림
## 사용법: python extract_streams1.py -f {파일명}.pcap -o {파일명}.csv
import argparse
import csv
import numpy as np
from collections import defaultdict
from scapy.all import PcapReader, Dot11
from scipy.stats import skew, kurtosis, entropy as scipy_entropy


def max_consecutive_packets(lengths, target=1090, tol=50):
    max_count = count = 0
    for l in lengths:
        if abs(l - target) <= tol:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
    return max_count


def safe_div(a, b):
    """0으로 나누는 경우를 방지한 나눗셈"""
    return a / b if b != 0 else 0.0


def compute_entropy(arr, bins=10):
    """
    주어진 1D 배열(arr)을 히스토그램으로 나눈 뒤,
    각 빈(bin)의 빈도수를 이용해 Shannon 엔트로피를 계산
    (히스토그램 빈도가 모두 0인 경우 엔트로피 0 반환).
    """
    if len(arr) == 0:
        return 0.0
    counts, _ = np.histogram(arr, bins=bins, density=False)
    probs = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)
    # 작은 값 더해서 log(0) 방지
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def stream_features(stream):
    """
    stream: {
      "packet_lengths": [...],
      "packet_intervals": [...],
      "timestamps": [...]
    }
    위 3가지 리스트를 받아 아래 다양한 특성을 계산하여 리스트로 반환.
    """
    lengths = np.array(stream["packet_lengths"], dtype=float)
    intervals = np.array(stream["packet_intervals"], dtype=float)
    timestamps = np.array(stream["timestamps"], dtype=float)

    total_packets = len(lengths)
    inbound_bytes = lengths.sum()
    length_1090_count = int(np.sum(np.abs(lengths - 1090) <= 50))

    # --- 기본 통계 (기존) ---
    mean_interval = float(np.mean(intervals)) if intervals.size > 0 else 0.0
    std_interval = float(np.std(intervals)) if intervals.size > 0 else 0.0
    mean_length = float(np.mean(lengths)) if lengths.size > 0 else 0.0
    std_length = float(np.std(lengths)) if lengths.size > 0 else 0.0
    max_consec_1090 = int(max_consecutive_packets(lengths.tolist()))

    # --- 시간(Time) 기반 특성 ---
    if timestamps.size >= 2:
        duration = float(timestamps.max() - timestamps.min())
    else:
        duration = 0.0
    packet_rate = safe_div(total_packets, duration)
    byte_rate = safe_div(inbound_bytes, duration)

    # --- 간격(Interval) 분포 특성 ---
    if intervals.size > 0:
        min_interval = float(np.min(intervals))
        max_interval = float(np.max(intervals))
        median_interval = float(np.median(intervals))
        burstiness = (
            safe_div((max_interval - min_interval), mean_interval)
            if mean_interval > 0
            else 0.0
        )
        # 엔트로피, 스큐, 첨도
        interval_entropy = compute_entropy(intervals, bins=10)
        interval_skew = float(skew(intervals)) if intervals.size > 2 else 0.0
        interval_kurt = float(kurtosis(intervals)) if intervals.size > 3 else 0.0
        cv_interval = safe_div(std_interval, mean_interval)
    else:
        min_interval = max_interval = median_interval = burstiness = 0.0
        interval_entropy = interval_skew = interval_kurt = cv_interval = 0.0

    # --- 패킷 크기(Size) 분포 특성 ---
    if lengths.size > 0:
        length_min = float(np.min(lengths))
        length_max = float(np.max(lengths))
        length_median = float(np.median(lengths))
        length_percentile25 = float(np.percentile(lengths, 25))
        length_percentile50 = length_median
        length_percentile75 = float(np.percentile(lengths, 75))
        length_entropy = compute_entropy(lengths, bins=10)
        length_skew = float(skew(lengths)) if lengths.size > 2 else 0.0
        length_kurt = float(kurtosis(lengths)) if lengths.size > 3 else 0.0
        unique_packet_sizes = int(len(np.unique(lengths)))
        num_large_pkts = int(np.sum(lengths >= 1000))
        # 1090 비율(정규화)
        pct_1090 = safe_div(length_1090_count, total_packets)
        # 작은 패킷(<100B), 큰 패킷(>800B) 비율
        small_packet_ratio = safe_div(np.sum(lengths < 100), total_packets)
        large_packet_ratio = safe_div(np.sum(lengths > 800), total_packets)
        cv_length = safe_div(std_length, mean_length)
    else:
        length_min = length_max = length_median = 0.0
        length_percentile25 = length_percentile50 = length_percentile75 = 0.0
        length_entropy = length_skew = length_kurt = 0.0
        unique_packet_sizes = num_large_pkts = 0
        pct_1090 = small_packet_ratio = large_packet_ratio = cv_length = 0.0

    # --- 기타 (예: 패킷 크기 대비 변화량 특성 등) ---
    # 여기서는 시퀀스 기반 변별 특성 예시로 RLE(run-length encoding) 정보를 구함
    #   연속해서 1090 근처 패킷이 몇 번씩 반복되는지 구함
    rle_runs = []
    count = 0
    for l in lengths:
        if abs(l - 1090) <= 50:
            count += 1
        else:
            if count > 0:
                rle_runs.append(count)
            count = 0
    if count > 0:
        rle_runs.append(count)
    rle_max = int(max(rle_runs)) if rle_runs else 0
    rle_mean = float(np.mean(rle_runs)) if rle_runs else 0.0

    # --- 결과 리스트로 묶기 ---
    features = [
        total_packets,  # 0
        length_1090_count,  # 1
        max_consec_1090,  # 2
        mean_interval,  # 3
        std_interval,  # 4
        mean_length,  # 5
        std_length,  # 6
        inbound_bytes,  # 7
        # 추가된 특성들:
        duration,  # 8
        packet_rate,  # 9
        byte_rate,  # 10
        min_interval,  # 11
        max_interval,  # 12
        median_interval,  # 13
        burstiness,  # 14
        interval_entropy,  # 15
        interval_skew,  # 16
        interval_kurt,  # 17
        cv_interval,  # 18
        length_min,  # 19
        length_max,  # 20
        length_median,  # 21
        length_percentile25,  # 22
        length_percentile50,  # 23
        length_percentile75,  # 24
        length_entropy,  # 25
        length_skew,  # 26
        length_kurt,  # 27
        unique_packet_sizes,  # 28
        num_large_pkts,  # 29
        pct_1090,  # 30
        small_packet_ratio,  # 31
        large_packet_ratio,  # 32
        cv_length,  # 33
        rle_max,  # 34
        rle_mean,  # 35
    ]

    return features


def cmd_extract(args):
    reader = PcapReader(args.pcap)

    # key를 (src_mac, dest_mac) 튜플로 사용
    stream_data = defaultdict(
        lambda: {"packet_lengths": [], "packet_intervals": [], "timestamps": []}
    )
    last_time = {}

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
        now = float(pkt.time)

        stream = stream_data[key]
        stream["packet_lengths"].append(plen)
        stream["timestamps"].append(now)

        if key in last_time:
            interval = now - last_time[key]
            stream["packet_intervals"].append(interval)
        last_time[key] = now

    reader.close()

    # CSV 헤더 정의 (위 features 순서에 맞춰 열 이름 모두 나열)
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
        "duration",
        "packet_rate",
        "byte_rate",
        "min_interval",
        "max_interval",
        "median_interval",
        "burstiness",
        "interval_entropy",
        "interval_skew",
        "interval_kurt",
        "cv_interval",
        "length_min",
        "length_max",
        "length_median",
        "length_percentile25",
        "length_percentile50",
        "length_percentile75",
        "length_entropy",
        "length_skew",
        "length_kurt",
        "unique_packet_sizes",
        "num_large_pkts",
        "pct_1090",
        "small_packet_ratio",
        "large_packet_ratio",
        "cv_length",
        "rle_max",
        "rle_mean",
    ]

    with open(args.output, "w", newline="") as wf:
        writer = csv.writer(wf)
        writer.writerow(header)

        for (src_mac, dest_mac), stream in stream_data.items():
            features = stream_features(stream)
            writer.writerow([src_mac, dest_mac] + features)

    print(f"Extracted {len(stream_data)} unique (src, dest) streams")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Wi-Fi stream features by (src, dest) MAC from a single PCAP"
    )
    parser.add_argument("-f", "--pcap", required=True, help="PCAP 파일 경로")
    parser.add_argument("-o", "--output", required=True, help="결과 CSV 파일 경로")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cmd_extract(args)
