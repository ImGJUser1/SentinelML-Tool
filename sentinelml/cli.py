import argparse
import pandas as pd
from .core import Sentinel

def scan_dataset(path):
    df = pd.read_csv(path)
    X = df.values

    sentinel = Sentinel()
    sentinel.fit(X)

    print("\nScanning dataset...\n")

    for i, row in enumerate(X[:10]):
        result = sentinel.assess(row)
        print(f"Row {i}: Trust={result['trust']:.3f}, Drift={result['drift_detected']}")

    print("\nDone.")

def main():
    parser = argparse.ArgumentParser(prog="sentinel")
    sub = parser.add_subparsers(dest="command")

    scan = sub.add_parser("scan")
    scan.add_argument("file")

    args = parser.parse_args()

    if args.command == "scan":
        scan_dataset(args.file)
    else:
        parser.print_help()