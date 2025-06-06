import os
from scapy.all import PcapReader, Dot11
import matplotlib.pyplot as plt
import numpy as np


def get_stream_stats(pkts):
    times = [float(pkt.time) for pkt in pkts]
    sizes = [len(pkt) for pkt in pkts]
    if len(times) > 1:
        mean_interval = np.mean(np.diff(sorted(times)))
    else:
        mean_interval = 0.0
    inbound_bytes = sum(sizes)  # 그냥 전체 bytes (inbound 개념 필요하면 조건 추가)
    return inbound_bytes, mean_interval, len(pkts)


def plot_streams(pcap_file, stream_keys, stream_pkt_map, title):
    plt.figure(figsize=(10, 6))
    for idx, key in enumerate(stream_keys):
        pkts = stream_pkt_map[key]
        times = [pkt.time for pkt in pkts]
        sizes = [len(pkt) for pkt in pkts]
        if not times:
            continue
        t0 = times[0]
        rel_times = [t - t0 for t in times]
        plt.scatter(
            rel_times, sizes, s=10, alpha=0.6, label=f"{idx+1}: {key[0]}→{key[1]}"
        )
    plt.xlabel("Time (s, relative)")
    plt.ylabel("Packet Length (bytes)")
    plt.title(title)
    plt.legend(fontsize=7, loc="best", ncol=2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pcap_file = "capture/test/dump_20250529_135053-01.pcap"

    from collections import defaultdict

    stream_pkt_map = defaultdict(list)
    reader = PcapReader(pcap_file)
    for pkt in reader:
        if not pkt.haslayer(Dot11):
            continue
        dot11 = pkt[Dot11]
        if dot11.type != 2:
            continue
        src = dot11.addr2
        dst = dot11.addr1
        if src is None or dst is None:
            continue
        key = (src, dst)
        stream_pkt_map[key].append(pkt)
    reader.close()

    stats = {}
    for key, pkts in stream_pkt_map.items():
        inbound_bytes, mean_interval, num_pkts = get_stream_stats(pkts)
        stats[key] = {
            "inbound_bytes": inbound_bytes,
            "mean_interval": mean_interval,
            "num_pkts": num_pkts,
        }

    # 각 조건별로 스트림 키 추출
    cond1_keys = [k for k, v in stats.items() if v["inbound_bytes"] >= 90000]
    cond2_keys = [k for k, v in stats.items() if v["mean_interval"] <= 1.0]
    cond3_keys = [k for k, v in stats.items() if v["num_pkts"] <= 1000]

    # 그래프 3개 그리기
    plot_streams(pcap_file, cond1_keys, stream_pkt_map, "inbound_bytes >= 90000")
    plot_streams(pcap_file, cond2_keys, stream_pkt_map, "mean_interval <= 1.0")
    plot_streams(pcap_file, cond3_keys, stream_pkt_map, "len(pkts) <= 1000")
