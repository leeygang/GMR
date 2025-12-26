#!/usr/bin/env python3
"""Batch convert GMR retargeted motions to AMP format and merge.

This script:
1. Converts all GMR .pkl files to AMP 29-dim format
2. Merges all AMP motions into a single dataset for training

v0.9.3: Added FK-based contact estimation support.
Joint order is loaded from robot_config.yaml (no hardcoding).

Usage:
    # Legacy (hip-pitch heuristic)
    cd ~/projects/GMR
    uv run python scripts/batch_convert_to_amp.py \
        --robot-config ~/projects/wildrobot/assets/robot_config.yaml

    # With FK-based contacts (recommended)
    cd ~/projects/GMR
    uv run python scripts/batch_convert_to_amp.py \
        --robot-config ~/projects/wildrobot/assets/robot_config.yaml \
        --robot-model ~/projects/wildrobot/assets/scene_flat_terrain.xml
"""

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Import from existing script
from convert_to_amp_format import (
    convert_to_amp_format,
    get_joint_names,
    load_mujoco_model,
    load_robot_config,
)
from rich import print
from rich.table import Table


def get_motion_duration(motion_path: Path) -> float:
    """Get duration of a motion file in seconds."""
    with open(motion_path, "rb") as f:
        data = pickle.load(f)
    return data.get("duration_sec", data["num_frames"] / data["fps"])


def batch_convert_to_amp(
    input_dir: Path,
    output_dir: Path,
    target_fps: float = 50.0,
    mj_model: Optional[Any] = None,
    contact_height_threshold: float = 0.02,
) -> List[Path]:
    """Convert all GMR motions in input_dir to AMP format.

    Args:
        input_dir: Directory containing GMR .pkl files
        output_dir: Directory for AMP output files
        target_fps: Target FPS for AMP format
        mj_model: Optional MuJoCo model for FK-based contact estimation
        contact_height_threshold: Foot height threshold for contact (m)

    Returns:
        List of successfully converted AMP file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    motion_files = sorted(input_dir.glob("*.pkl"))
    converted = []

    contact_method = "FK" if mj_model is not None else "heuristic"
    print(
        f"\n[bold blue]Converting {len(motion_files)} motions to AMP format ({contact_method} contacts)...[/bold blue]"
    )

    for motion_path in motion_files:
        output_path = output_dir / f"{motion_path.stem}_amp.pkl"

        # Skip if already converted
        if output_path.exists():
            print(f"  [dim]SKIP[/dim] {motion_path.name} (already exists)")
            converted.append(output_path)
            continue

        try:
            # Load GMR motion
            with open(motion_path, "rb") as f:
                motion_data = pickle.load(f)

            # Convert to AMP format (with optional FK-based contacts)
            amp_data = convert_to_amp_format(
                motion_data,
                target_fps=target_fps,
                mj_model=mj_model,
                contact_height_threshold=contact_height_threshold,
            )

            # Save AMP format
            with open(output_path, "wb") as f:
                pickle.dump(amp_data, f)

            print(
                f"  [green]✓[/green] {motion_path.name} -> {output_path.name} ({amp_data['duration_sec']:.2f}s, {amp_data['num_frames']} frames)"
            )
            converted.append(output_path)

        except Exception as e:
            print(f"  [red]✗[/red] {motion_path.name}: {e}")

    return converted


def merge_amp_motions(
    amp_files: List[Path],
    output_path: Path,
) -> Dict[str, Any]:
    """Merge multiple AMP motion files into a single dataset.

    The merged dataset contains all frames from all motions concatenated together.
    This is the format expected by the AMP discriminator for sampling reference frames.

    Args:
        amp_files: List of AMP .pkl file paths
        output_path: Output path for merged dataset

    Returns:
        Merged dataset dictionary
    """
    all_features = []
    all_dof_pos = []
    all_dof_vel = []
    source_files = []
    total_duration = 0.0

    print(f"\n[bold blue]Merging {len(amp_files)} AMP motions...[/bold blue]")

    for amp_path in amp_files:
        with open(amp_path, "rb") as f:
            data = pickle.load(f)

        all_features.append(data["features"])
        all_dof_pos.append(data["dof_pos"])
        all_dof_vel.append(data["dof_vel"])
        source_files.append(str(amp_path.name))
        total_duration += data["duration_sec"]

        print(
            f"  + {amp_path.name}: {data['num_frames']} frames, {data['duration_sec']:.2f}s"
        )

    # Concatenate all arrays
    merged = {
        # Main AMP features for discriminator
        "features": np.concatenate(all_features, axis=0).astype(np.float32),
        "feature_dim": 29,
        # Additional data for debugging/analysis
        "dof_pos": np.concatenate(all_dof_pos, axis=0).astype(np.float32),
        "dof_vel": np.concatenate(all_dof_vel, axis=0).astype(np.float32),
        # Metadata
        "fps": 50.0,
        "dt": 0.02,
        "num_frames": sum(len(f) for f in all_features),
        "duration_sec": total_duration,
        "source_files": source_files,
        "num_motions": len(amp_files),
        # Joint info from robot_config.yaml (no hardcoding)
        "joint_names": get_joint_names(),
    }

    # Save merged dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(merged, f)

    print(f"\n[bold green]Merged dataset saved to:[/bold green] {output_path}")

    return merged


def print_dataset_summary(merged: Dict[str, Any]):
    """Print summary statistics of the merged dataset."""
    features = merged["features"]

    table = Table(title="Merged AMP Dataset Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Frames", f"{merged['num_frames']:,}")
    table.add_row("Total Duration", f"{merged['duration_sec']:.2f}s")
    table.add_row("Number of Motions", str(merged["num_motions"]))
    table.add_row("Feature Dimension", str(merged["feature_dim"]))
    table.add_row("FPS", str(merged["fps"]))
    table.add_row("Features Shape", str(features.shape))

    print(table)

    # Per-component statistics
    labels = [
        ("Joint Positions (0-8)", 0, 9),
        ("Joint Velocities (9-17)", 9, 18),
        ("Root Lin Vel (18-20)", 18, 21),
        ("Root Ang Vel (21-23)", 21, 24),
        ("Root Height (24)", 24, 25),
        ("Foot Contacts (25-28)", 25, 29),
    ]

    stats_table = Table(title="Feature Statistics")
    stats_table.add_column("Component", style="cyan")
    stats_table.add_column("Min", style="yellow")
    stats_table.add_column("Max", style="yellow")
    stats_table.add_column("Mean", style="green")
    stats_table.add_column("Std", style="blue")

    for label, start, end in labels:
        component = features[:, start:end]
        stats_table.add_row(
            label,
            f"{component.min():.3f}",
            f"{component.max():.3f}",
            f"{component.mean():.3f}",
            f"{component.std():.3f}",
        )

    print(stats_table)


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert GMR motions to AMP format and merge"
    )
    parser.add_argument(
        "--robot-config",
        type=str,
        required=True,
        help="Path to robot_config.yaml (required for joint order)",
    )
    parser.add_argument(
        "--robot-model",
        type=str,
        default=None,
        help="Path to MuJoCo XML model for FK-based contact estimation (v0.9.3)",
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=0.02,
        help="Foot height threshold for contact detection (m), default=0.02",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(Path.home() / "projects/wildrobot/assets/motions"),
        help="Directory containing GMR .pkl files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path.home() / "projects/wildrobot/playground_amp/data"),
        help="Directory for AMP output files",
    )
    parser.add_argument(
        "--merged-output",
        type=str,
        default="walking_motions_merged.pkl",
        help="Filename for merged dataset (in output-dir)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=50.0,
        help="Target FPS for AMP format",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reconversion even if output files exist",
    )

    args = parser.parse_args()

    # Load robot config first (required for joint order)
    print(f"[bold blue]Loading robot config:[/bold blue] {args.robot_config}")
    config = load_robot_config(args.robot_config)
    joint_names = config["actuators"]["joints"]
    print(f"  Joint order: {joint_names}")

    # Load MuJoCo model if provided (for FK-based contacts)
    mj_model = None
    if args.robot_model:
        print(f"[bold blue]Loading MuJoCo model:[/bold blue] {args.robot_model}")
        mj_model = load_mujoco_model(args.robot_model)
        if mj_model is not None:
            print(f"  [green]✓ Model loaded successfully[/green]")
        else:
            print(
                f"  [yellow]⚠ Failed to load model, will use heuristic contacts[/yellow]"
            )

    # Paths
    gmr_motion_dir = Path(args.input_dir)
    amp_output_dir = Path(args.output_dir)
    merged_output = amp_output_dir / args.merged_output

    # Handle --force: remove existing AMP files to force reconversion
    if args.force:
        print("[yellow]--force: Removing existing AMP files for reconversion[/yellow]")
        for amp_file in amp_output_dir.glob("*_amp.pkl"):
            amp_file.unlink()
            print(f"  [dim]Removed: {amp_file.name}[/dim]")

    contact_method = "FK" if mj_model else "heuristic"
    print(f"{'='*60}")
    print("Batch Convert GMR Motions to AMP Format & Merge")
    print(f"{'='*60}")
    print(f"Input:  {gmr_motion_dir}")
    print(f"Output: {amp_output_dir}")
    print(f"Merged: {merged_output}")
    print(f"Contact: {contact_method}")
    print(f"{'='*60}")

    # Step 1: Convert all GMR motions to AMP format
    amp_files = batch_convert_to_amp(
        input_dir=gmr_motion_dir,
        output_dir=amp_output_dir,
        target_fps=args.target_fps,
        mj_model=mj_model,
        contact_height_threshold=args.contact_threshold,
    )

    if not amp_files:
        print("[red]No AMP files to merge![/red]")
        return

    # Step 2: Merge all AMP motions into single dataset
    merged = merge_amp_motions(amp_files, merged_output)

    # Step 3: Print summary
    print_dataset_summary(merged)

    print(f"\n[bold green]✓ Done! Use this file for AMP training:[/bold green]")
    print(f"  {merged_output}")
    print(f"\n[dim]Example usage:[/dim]")
    print(f"  python playground_amp/train_amp.py --amp-data {merged_output}")


if __name__ == "__main__":
    main()
