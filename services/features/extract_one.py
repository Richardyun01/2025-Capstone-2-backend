import glob
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PCAP_EXTRACT_PATH = os.path.join(BASE_DIR, "pcap_extract.py")


def extract_one(pcap_path, output_dir="features", labels_dir="labels", window=11):
    base = os.path.splitext(os.path.basename(pcap_path))[0]
    feat_file = os.path.join(output_dir, f"{base}.csv")
    lbl_file = os.path.join(labels_dir, f"{base}.lbl")
    try:
        subprocess.run(
            [
                "python",
                PCAP_EXTRACT_PATH,
                "-f",
                pcap_path,
                "-o",
                feat_file,
                "-l",
                lbl_file,
            ],
            check=True,
        )
        print(f"[SUCCESS] Started processing {pcap_path}")
    except subprocess.CalledProcessError:
        print(f"[FAILURE] extract failed for {pcap_path}")


if __name__ == "__main__":
    os.makedirs("features", exist_ok=True)
    os.makedirs("labels", exist_ok=True)
    print(f"Starting extraction for {os.getcwd()}")

    pcaps = glob.glob("capture/**/*.pcap", recursive=True)
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(extract_one, pcaps)
