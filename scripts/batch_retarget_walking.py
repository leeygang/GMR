#!/usr/bin/env python3
"""Batch retarget walking motions from AMASS KIT dataset to WildRobot.

This script retargets multiple walking motions to expand the reference motion
dataset for AMP training. Target: 50+ seconds of walking data.

Usage:
    cd ~/projects/GMR

    # Default output (assets/motions)
    uv run python scripts/batch_retarget_walking.py

    # Custom output directory
    uv run python scripts/batch_retarget_walking.py --output-dir ~/projects/wildrobot/playground_amp/data/gmr

    # Force regenerate (overwrite existing)
    uv run python scripts/batch_retarget_walking.py --force
"""

import argparse
import subprocess
import sys
from pathlib import Path


# Walking motions to retarget
# Format: (source_path_relative_to_KIT, output_name)
# Selected for variety: different speeds (slow/medium/fast) and different subjects
WALKING_MOTIONS = [
    # Subject 3 - various speeds
    ("3/walking_slow01_stageii.npz", "walking_slow01"),
    ("3/walking_slow02_stageii.npz", "walking_slow02"),
    ("3/walking_medium01_stageii.npz", "walking_medium01"),
    ("3/walking_medium02_stageii.npz", "walking_medium02"),
    ("3/walking_fast01_stageii.npz", "walking_fast01"),
    ("3/walking_fast02_stageii.npz", "walking_fast02"),

    # Subject 167 - different person
    ("167/walking_slow01_stageii.npz", "walking_slow03"),
    ("167/walking_medium01_stageii.npz", "walking_medium03"),
    ("167/walking_fast01_stageii.npz", "walking_fast03"),

    # Subject 183 - another person
    ("183/walking_slow01_stageii.npz", "walking_slow04"),
    ("183/walking_medium01_stageii.npz", "walking_medium04"),

    # Subject 317 - for more variety
    ("317/walking_slow01_stageii.npz", "walking_slow05"),
    ("317/walking_medium01_stageii.npz", "walking_medium05"),

    # Subject 359 - another style
    ("359/walking_slow01_stageii.npz", "walking_slow06"),
    ("359/walking_medium01_stageii.npz", "walking_medium06"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Batch retarget walking motions from AMASS KIT to WildRobot"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path.home() / "projects/wildrobot/assets/motions"),
        help="Output directory for retargeted motion files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regenerate (overwrite existing files)",
    )
    args = parser.parse_args()

    amass_base = Path.home() / "projects/amass/smplx/KIT"
    output_base = Path(args.output_dir)
    gmr_dir = Path.home() / "projects/GMR"

    # Ensure output directory exists
    output_base.mkdir(parents=True, exist_ok=True)

    total_retargeted = 0
    failed = []

    print(f"{'='*60}")
    print("Batch Retargeting Walking Motions from AMASS KIT to WildRobot")
    print(f"{'='*60}")
    print(f"Source: {amass_base}")
    print(f"Output: {output_base}")
    print(f"Motions to process: {len(WALKING_MOTIONS)}")
    print(f"Force overwrite: {args.force}")
    print(f"{'='*60}\n")

    for source_rel, output_name in WALKING_MOTIONS:
        source_path = amass_base / source_rel
        output_path = output_base / f"{output_name}.pkl"

        # Skip if output already exists (unless --force)
        if output_path.exists() and not args.force:
            print(f"[SKIP] {output_name} - already exists")
            continue

        # Check if source exists
        if not source_path.exists():
            print(f"[WARN] {source_rel} - source not found, skipping")
            failed.append((source_rel, "source not found"))
            continue

        print(f"[PROCESSING] {source_rel} -> {output_name}.pkl")

        # Run retargeting
        cmd = [
            "uv", "run", "python", "scripts/smplx_to_robot_headless.py",
            "--smplx_file", str(source_path),
            "--robot", "wildrobot",
            "--save_path", str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=gmr_dir,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout per motion
            )

            if result.returncode == 0:
                print(f"  ✓ Success: {output_name}.pkl")
                total_retargeted += 1
            else:
                print(f"  ✗ Failed: {result.stderr[:200]}")
                failed.append((source_rel, result.stderr[:100]))

        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout")
            failed.append((source_rel, "timeout"))
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed.append((source_rel, str(e)))

    print(f"\n{'='*60}")
    print(f"Completed: {total_retargeted} motions")
    if failed:
        print(f"Failed: {len(failed)} motions")
        for name, reason in failed:
            print(f"  - {name}: {reason[:50]}")
    print(f"{'='*60}")

    # Check total duration
    print("\nChecking total motion duration...")
    import pickle
    total_duration = 0
    for f in output_base.glob("*.pkl"):
        try:
            with open(f, 'rb') as fp:
                data = pickle.load(fp)
            dur = data.get('duration_sec', data['num_frames'] / data['fps'])
            total_duration += dur
        except:
            pass

    print(f"Total duration: {total_duration:.2f}s")
    if total_duration >= 50:
        print("✓ Target of 50s+ achieved!")
    else:
        print(f"Need {50 - total_duration:.2f}s more")


if __name__ == "__main__":
    main()
