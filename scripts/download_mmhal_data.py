#!/usr/bin/env python
"""
Download MMHal-Bench dataset template.

Run this first to download the dataset before evaluation.

Usage:
    python scripts/download_mmhal_data.py
"""

import requests
import json
from pathlib import Path


def main():
    print("Downloading MMHal-Bench template...")

    template_path = Path("data/mmhal_bench_response_template.json")
    template_path.parent.mkdir(parents=True, exist_ok=True)

    # Download from HuggingFace
    url = "https://huggingface.co/datasets/Shengcao1006/MMHal-Bench/resolve/main/response_template.json"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Validate JSON
        data = response.json()
        print(f"[OK] Downloaded {len(data)} samples")

        # Save to file
        with open(template_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved to {template_path}")
        print("\nDataset ready! You can now run:")
        print("  python scripts/evaluate_mmhal_baseline.py --debug")

    except Exception as e:
        print(f"[ERROR] {e}")
        print("\nAlternative: Download manually from:")
        print("https://huggingface.co/datasets/Shengcao1006/MMHal-Bench/blob/main/response_template.json")
        print(f"Save to: {template_path.absolute()}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
