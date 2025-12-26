#!/usr/bin/env python3
"""Headless SMPLX to robot retargeting script (no visualization).

This script runs GMR retargeting without launching the MuJoCo viewer,
which is required for running on macOS without mjpython.

Usage:
    uv run python scripts/smplx_to_robot_headless.py \
        --smplx_file /path/to/motion.npz \
        --robot wildrobot \
        --save_path output.pkl
"""

import argparse
import os
import pathlib
import pickle
import time

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import (
    get_smplx_data_offline_fast,
    load_smplx_file,
)
from rich import print


if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser(description="Headless SMPLX to robot retargeting")
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_g1_with_hands",
            "unitree_h1",
            "unitree_h1_2",
            "booster_t1",
            "booster_t1_29dof",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
            "kuavo_s45",
            "hightorque_hi",
            "galaxea_r1pro",
            "berkeley_humanoid_lite",
            "booster_k1",
            "pnd_adam_lite",
            "openloong",
            "tienkung",
            "pal_talos",
            "wildrobot",
        ],
        default="wildrobot",
    )
    parser.add_argument(
        "--save_path",
        default="retargeted_motion.pkl",
        help="Path to save the robot motion.",
    )
    parser.add_argument(
        "--tgt_fps",
        type=int,
        default=30,
        help="Target FPS for the retargeted motion.",
    )
    parser.add_argument(
        "--ik_config",
        type=str,
        default=None,
        help="Path to IK config JSON file (default: use robot's default config)",
    )

    args = parser.parse_args()

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

    print(f"[bold blue]Loading SMPLX file:[/bold blue] {args.smplx_file}")

    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    print(f"[green]Human height:[/green] {actual_human_height:.2f}m")

    # Align FPS
    tgt_fps = args.tgt_fps
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )
    print(f"[green]Frames:[/green] {len(smplx_data_frames)} at {aligned_fps} FPS")

    # Initialize the retargeting system
    print(f"[bold blue]Initializing GMR for robot:[/bold blue] {args.robot}")
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
        ik_config_path=args.ik_config,
    )

    # Run retargeting
    print(f"[bold blue]Retargeting {len(smplx_data_frames)} frames...[/bold blue]")
    qpos_list = []

    start_time = time.time()
    for i, smplx_data in enumerate(smplx_data_frames):
        qpos = retarget.retarget(smplx_data)
        qpos_list.append(qpos)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            fps = (i + 1) / elapsed
            print(f"  Frame {i+1}/{len(smplx_data_frames)} ({fps:.1f} fps)")

    elapsed = time.time() - start_time
    print(
        f"[green]Retargeting complete:[/green] {len(qpos_list)} frames in {elapsed:.2f}s ({len(qpos_list)/elapsed:.1f} fps)"
    )

    # Save results
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    root_pos = np.array([qpos[:3] for qpos in qpos_list])
    # Convert from wxyz to xyzw quaternion format
    root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])

    motion_data = {
        "fps": aligned_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
        "source_file": args.smplx_file,
        "robot": args.robot,
        "num_frames": len(qpos_list),
        "duration_sec": len(qpos_list) / aligned_fps,
    }

    with open(args.save_path, "wb") as f:
        pickle.dump(motion_data, f)

    print(f"[bold green]Saved to:[/bold green] {args.save_path}")
    print(f"  Frames: {motion_data['num_frames']}")
    print(f"  Duration: {motion_data['duration_sec']:.2f}s")
    print(f"  DOF shape: {dof_pos.shape}")
