import argparse, csv
import numpy as np
from collections import defaultdict
from scapy.all import PcapReader, Dot11


def cmd_extract(args):
    reader = PcapReader(args.pcap)
    stream_dict = defaultdict(list)

    for pkt in reader:
        if not pkt.haslayer(Dot11):
            continue
        dot11 = pkt[Dot11]
        if dot11.type != 2:
            continue

        src_mac, dst_mac = dot11.addr2, dot11.addr1
        if src_mac is None or dst_mac is None:
            continue

        key = (src_mac, dst_mac)
        stream_dict[key].append(pkt)
    reader.close()

    with open(args.output, "w", newline="") as wf, open(args.labels, "w") as lf:
        writer = csv.writer(wf)
        writer.writerow(
            [
                "stream_key",
                "total_inbound",
                "total_outbound",
                "mean_interval",
                "std_interval",
                "pkt_count",
            ]
        )

        for (src, dst), pkts in stream_dict.items():
            inbound_bytes = outbound_bytes = 0
            last_time, intervals = None, []

            for pkt in pkts:
                plen = len(pkt)
                if pkt[Dot11].addr1 == src:
                    outbound_bytes += plen
                else:
                    inbound_bytes += plen

                if last_time is not None:
                    intervals.append(float(pkt.time) - float(last_time))
                last_time = pkt.time

            mean_interval = np.mean(intervals) if intervals else 0
            std_interval = np.std(intervals) if intervals else 0

            stream_key = f"{src}|{dst}"
            writer.writerow(
                [
                    stream_key,
                    inbound_bytes,
                    outbound_bytes,
                    mean_interval,
                    std_interval,
                    len(pkts),
                ]
            )

            is_camera = (
                inbound_bytes >= 90000 and mean_interval <= 1.0 and len(pkts) <= 1000
            )
            lf.write(f"{int(is_camera)}\n")

    print(f"Extracted {len(stream_dict)} streams")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Wi-Fi stream features")
    parser.add_argument("-f", "--pcap", required=True, help="PCAP file path")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file")
    parser.add_argument("-l", "--labels", required=True, help="Output labels file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cmd_extract(args)
