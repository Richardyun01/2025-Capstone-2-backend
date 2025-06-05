## src 기반 스트림
## 사용법: python extract_streams2.py -f {파일명1}.pcap -o {파일명2}.csv
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


def stream_features(stream):
    lengths = stream["packet_lengths"]
    intervals = stream["packet_intervals"]

    length_1090_count = sum(1 for l in lengths if abs(l - 1090) <= 50)
    total_packets = len(lengths)
    inbound_bytes = sum(lengths)
    mean_interval = np.mean(intervals) if intervals else 0.0

    features = [
        total_packets,  # 0
        length_1090_count,  # 1
        max_consecutive_packets(lengths),  # 2
        mean_interval,  # 3
        np.std(intervals) if intervals else 0.0,  # 4
        np.mean(lengths) if lengths else 0.0,  # 5
        np.std(lengths) if lengths else 0.0,  # 6
        inbound_bytes,  # 7
    ]
    return features


def cmd_extract(args):
    reader = PcapReader(args.pcap)
    stream_data = defaultdict(
        lambda: {"packet_lengths": [], "packet_intervals": [], "timestamps": []}
    )
    last_time = {}

    for pkt in reader:
        if not pkt.haslayer(Dot11):
            continue
        dot11 = pkt[Dot11]
        # data frame (type=2)이며 addr2(src)가 있어야 함
        if dot11.type != 2 or dot11.addr2 is None:
            continue

        src_mac = dot11.addr2
        key = src_mac

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

    # CSV 헤더에 src_mac만 남기고, feature 컬럼들 기록
    with open(args.output, "w", newline="") as wf:
        writer = csv.writer(wf)
        writer.writerow(
            [
                "src_mac",
                "total_packets",
                "length_1090_count",
                "max_consecutive_1090",
                "mean_interval",
                "std_interval",
                "mean_length",
                "std_length",
                "inbound_bytes",
            ]
        )

        for src_mac, stream in stream_data.items():
            features = stream_features(stream)
            writer.writerow([src_mac] + features)

    print(f"Extracted {len(stream_data)} unique src_mac streams")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Wi-Fi stream features grouped by src MAC from a single PCAP"
    )
    parser.add_argument("-f", "--pcap", required=True, help="PCAP 파일 경로")
    parser.add_argument("-o", "--output", required=True, help="결과 CSV 파일 경로")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cmd_extract(args)
