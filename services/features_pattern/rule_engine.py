import numpy as np


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
    return [
        len(lengths),  # 0 - total_packets
        sum(1 for l in lengths if abs(l - 1090) <= 50),  # 1 - length_1090_count
        max_consecutive_packets(lengths),  # 2 - max_consecutive_1090
        np.mean(intervals) if intervals else 0,  # 3 - mean_interval
        np.std(intervals) if intervals else 0,  # 4 - std_interval
        np.mean(lengths),  # 5 - mean_length
        np.std(lengths),  # 6 - std_length
        sum(lengths),  # 7 - inbound_bytes
    ]


def label_stream(features):
    total_packets = features[0]
    length_1090_count = features[1]
    max_consec = features[2]
    mean_interval = features[3]
    mean_length = features[5]
    inbound_bytes = features[7]

    rule = (
        length_1090_count >= 50
        and max_consec >= 8
        and mean_interval <= 0.5
        and mean_length >= 600
        and total_packets >= 100
        and inbound_bytes >= 60000
    )

    return rule
