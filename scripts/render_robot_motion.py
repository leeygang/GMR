#!/usr/bin/env python3
"""Render robot motion to video using offline MuJoCo rendering (no interactive viewer).

Works on macOS without mjpython by using the offscreen renderer.

Usage:
    uv run python scripts/render_robot_motion.py \
        --robot wildrobot \
        --motion_path /path/to/motion.pkl \
        --output_video output.mp4
"""

import argparse
import pathlib
import pickle

import cv2
import mujoco
import numpy as np
from general_motion_retargeting.params import ROBOT_XML_DICT
from rich import print


def render_motion_to_video(
    robot_xml: str,
    motion_data: dict,
    output_path: str,
    width: int = 1280,
    height: int = 720,
    camera_distance: float = 2.0,
    camera_azimuth: float = 90.0,
    camera_elevation: float = -20.0,
):
    """Render robot motion to video using offscreen rendering."""

    # Load model
    model = mujoco.MjModel.from_xml_path(robot_xml)
    data = mujoco.MjData(model)

    # Setup renderer
    renderer = mujoco.Renderer(model, height, width)

    # Setup camera
    camera = mujoco.MjvCamera()
    camera.distance = camera_distance
    camera.azimuth = camera_azimuth
    camera.elevation = camera_elevation
    camera.lookat[:] = [0, 0, 0.5]

    # Setup video writer
    fps = int(motion_data.get("fps", 30))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    root_pos = motion_data["root_pos"]
    root_rot = motion_data["root_rot"]  # xyzw format
    dof_pos = motion_data["dof_pos"]

    num_frames = len(root_pos)
    print(f"Rendering {num_frames} frames at {fps} FPS...")

    for i in range(num_frames):
        # Set root position (first 3 elements of qpos)
        data.qpos[:3] = root_pos[i]

        # Set root orientation (wxyz format for MuJoCo)
        # Convert from xyzw to wxyz
        quat_xyzw = root_rot[i]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        data.qpos[3:7] = quat_wxyz

        # Set joint positions
        data.qpos[7 : 7 + len(dof_pos[i])] = dof_pos[i]

        # Forward kinematics
        mujoco.mj_forward(model, data)

        # Update camera to track robot
        camera.lookat[:] = data.qpos[:3]
        camera.lookat[2] = 0.5  # Keep looking at mid-height

        # Render
        renderer.update_scene(data, camera)
        frame = renderer.render()

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

        if (i + 1) % 50 == 0:
            print(f"  Frame {i+1}/{num_frames}")

    writer.release()
    print(f"[bold green]Video saved to:[/bold green] {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render robot motion to video")
    parser.add_argument(
        "--robot",
        type=str,
        default="wildrobot",
        help="Robot type",
    )
    parser.add_argument(
        "--motion_path",
        type=str,
        required=True,
        help="Path to motion pickle file",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default="output.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Video width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Video height",
    )
    parser.add_argument(
        "--camera_distance",
        type=float,
        default=2.0,
        help="Camera distance from robot",
    )

    args = parser.parse_args()

    # Get robot XML path
    robot_xml = str(ROBOT_XML_DICT[args.robot])
    print(f"[bold blue]Robot XML:[/bold blue] {robot_xml}")

    # Load motion data
    print(f"[bold blue]Loading motion:[/bold blue] {args.motion_path}")
    with open(args.motion_path, "rb") as f:
        motion_data = pickle.load(f)

    print(f"  Frames: {motion_data['num_frames']}")
    print(f"  Duration: {motion_data['duration_sec']:.2f}s")
    print(f"  FPS: {motion_data['fps']:.1f}")

    # Render video
    render_motion_to_video(
        robot_xml=robot_xml,
        motion_data=motion_data,
        output_path=args.output_video,
        width=args.width,
        height=args.height,
        camera_distance=args.camera_distance,
    )
