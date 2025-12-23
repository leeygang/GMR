#!/usr/bin/env python3
"""Convert GMR retargeted motion to AMP training format.

Converts the pickle output from GMR (smplx_to_robot_headless.py) to the format
needed for AMP discriminator training in WildRobot.

The AMP format includes:
- Joint positions (9 DOF)
- Joint velocities (9 DOF) - computed from positions
- Root linear velocity (3D)
- Root angular velocity (3D)
- Root height (1D)
- Foot contact estimates (4D - left/right toe/heel)

Total: 29 features per frame

Joint order is loaded from robot_config.yaml (no hardcoding).

Usage:
    uv run python scripts/convert_to_amp_format.py \
        --input /path/to/gmr_output.pkl \
        --output /path/to/amp_motion.pkl \
        --robot-config /path/to/robot_config.yaml \
        --target_fps 50
"""

import argparse
import pathlib
import pickle
from typing import Dict, Any, List

import numpy as np
import yaml
from rich import print


# Global robot config (loaded once)
_robot_config: Dict[str, Any] = None


def load_robot_config(config_path: str) -> Dict[str, Any]:
    """Load robot configuration from YAML file.

    Args:
        config_path: Path to robot_config.yaml

    Returns:
        Robot configuration dict
    """
    global _robot_config
    with open(config_path, 'r') as f:
        _robot_config = yaml.safe_load(f)
    return _robot_config


def get_robot_config() -> Dict[str, Any]:
    """Get cached robot configuration.

    Raises:
        RuntimeError: If config not loaded via load_robot_config()
    """
    if _robot_config is None:
        raise RuntimeError(
            "Robot config not loaded. Call load_robot_config() first "
            "or pass --robot-config argument."
        )
    return _robot_config


def get_joint_names() -> List[str]:
    """Get joint names in MuJoCo qpos order from robot config."""
    config = get_robot_config()
    return config['actuators']['joints']


def get_joint_idx(joint_name: str) -> int:
    """Get index of a joint by name.

    Args:
        joint_name: Name of the joint

    Returns:
        Index in the joint array

    Raises:
        ValueError: If joint not found
    """
    joint_names = get_joint_names()
    try:
        return joint_names.index(joint_name)
    except ValueError:
        raise ValueError(
            f"Joint '{joint_name}' not found in robot config. "
            f"Available joints: {joint_names}"
        )


def quaternion_to_angular_velocity(
    quats: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Compute angular velocity from quaternion time series.

    Args:
        quats: Quaternions in xyzw format, shape (N, 4)
        dt: Time step

    Returns:
        Angular velocities, shape (N, 3)
    """
    N = len(quats)
    ang_vel = np.zeros((N, 3), dtype=np.float32)

    for i in range(1, N):
        q0 = quats[i-1]  # xyzw
        q1 = quats[i]    # xyzw

        # Quaternion derivative: dq/dt ≈ (q1 - q0) / dt
        # Angular velocity: ω = 2 * conj(q) * dq/dt

        # Compute quaternion derivative
        dq = (q1 - q0) / dt

        # Conjugate of q0: negate xyz components
        q0_conj = np.array([-q0[0], -q0[1], -q0[2], q0[3]])

        # Quaternion multiply: q0_conj * dq
        # Result gives angular velocity as 2 * xyz components
        w0, x0, y0, z0 = q0_conj[3], q0_conj[0], q0_conj[1], q0_conj[2]
        w1, x1, y1, z1 = dq[3], dq[0], dq[1], dq[2]

        wx = w0*x1 + x0*w1 + y0*z1 - z0*y1
        wy = w0*y1 - x0*z1 + y0*w1 + z0*x1
        wz = w0*z1 + x0*y1 - y0*x1 + z0*w1

        ang_vel[i] = 2 * np.array([wx, wy, wz])

    ang_vel[0] = ang_vel[1] if N > 1 else 0
    return ang_vel


def resample_motion(motion_data: Dict[str, Any], target_fps: float) -> Dict[str, Any]:
    """Resample motion to target FPS using linear interpolation.

    Args:
        motion_data: Original motion data
        target_fps: Target frames per second

    Returns:
        Resampled motion data
    """
    source_fps = motion_data['fps']
    num_frames = motion_data['num_frames']
    duration = motion_data['duration_sec']

    # Calculate new number of frames
    new_num_frames = int(duration * target_fps)

    # Original and new time arrays
    t_original = np.linspace(0, duration, num_frames)
    t_new = np.linspace(0, duration, new_num_frames)

    # Interpolate each array
    def interp_array(arr):
        if arr is None:
            return None
        if arr.ndim == 1:
            return np.interp(t_new, t_original, arr).astype(np.float32)
        else:
            result = np.zeros((new_num_frames, arr.shape[1]), dtype=np.float32)
            for i in range(arr.shape[1]):
                result[:, i] = np.interp(t_new, t_original, arr[:, i])
            return result

    # Interpolate quaternions with normalization
    def interp_quat(quats):
        result = interp_array(quats)
        # Normalize quaternions
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        result = result / np.clip(norms, 1e-8, None)
        return result

    return {
        'fps': target_fps,
        'root_pos': interp_array(motion_data['root_pos']),
        'root_rot': interp_quat(motion_data['root_rot']),
        'dof_pos': interp_array(motion_data['dof_pos']),
        'source_file': motion_data.get('source_file', 'unknown'),
        'robot': motion_data.get('robot', 'wildrobot'),
        'num_frames': new_num_frames,
        'duration_sec': duration,
    }


def estimate_foot_contacts(
    dof_pos: np.ndarray,
    threshold_angle: float = 0.1,
) -> np.ndarray:
    """Estimate foot contacts from joint positions.

    Uses hip pitch angles to estimate gait phase.
    When hip pitch is negative (leg behind), foot is likely in contact.

    Joint indices are loaded from robot_config.yaml (no hardcoding).

    Args:
        dof_pos: Joint positions (N, num_joints)
        threshold_angle: Angle threshold for contact detection

    Returns:
        Foot contacts (N, 4) - [left_toe, left_heel, right_toe, right_heel]
    """
    N = len(dof_pos)
    contacts = np.zeros((N, 4), dtype=np.float32)

    # Get joint indices from config (no hardcoding)
    idx_left_hip_pitch = get_joint_idx('left_hip_pitch')
    idx_left_knee = get_joint_idx('left_knee_pitch')
    idx_right_hip_pitch = get_joint_idx('right_hip_pitch')
    idx_right_knee = get_joint_idx('right_knee_pitch')

    left_hip_pitch = dof_pos[:, idx_left_hip_pitch]
    right_hip_pitch = dof_pos[:, idx_right_hip_pitch]
    left_knee = dof_pos[:, idx_left_knee]
    right_knee = dof_pos[:, idx_right_knee]

    # Left foot contact (more confidence when leg behind + straight knee)
    left_contact = (left_hip_pitch < threshold_angle).astype(np.float32)
    left_contact *= np.clip(1.0 - np.abs(left_knee) / 0.5, 0.3, 1.0)

    # Right foot contact
    right_contact = (right_hip_pitch < threshold_angle).astype(np.float32)
    right_contact *= np.clip(1.0 - np.abs(right_knee) / 0.5, 0.3, 1.0)

    # Set toe and heel contacts (simplified - same value for both)
    contacts[:, 0] = left_contact  # left toe
    contacts[:, 1] = left_contact  # left heel
    contacts[:, 2] = right_contact  # right toe
    contacts[:, 3] = right_contact  # right heel

    return contacts


def convert_to_amp_format(
    motion_data: Dict[str, Any],
    target_fps: float = 50.0,
) -> Dict[str, Any]:
    """Convert GMR output to AMP training format.

    Args:
        motion_data: Output from GMR retargeting
        target_fps: Target FPS for AMP training

    Returns:
        AMP-formatted motion data
    """
    # Resample to target FPS if needed
    source_fps = motion_data['fps']
    if abs(source_fps - target_fps) > 0.1:
        print(f"[yellow]Resampling from {source_fps:.1f} FPS to {target_fps:.1f} FPS[/yellow]")
        motion_data = resample_motion(motion_data, target_fps)

    dt = 1.0 / target_fps
    num_frames = motion_data['num_frames']

    dof_pos = motion_data['dof_pos']
    root_pos = motion_data['root_pos']
    root_rot = motion_data['root_rot']  # xyzw format

    # Compute joint velocities
    dof_vel = np.zeros_like(dof_pos)
    dof_vel[1:] = (dof_pos[1:] - dof_pos[:-1]) / dt
    dof_vel[0] = dof_vel[1] if num_frames > 1 else 0

    # Compute root linear velocity
    root_lin_vel = np.zeros_like(root_pos)
    root_lin_vel[1:] = (root_pos[1:] - root_pos[:-1]) / dt
    root_lin_vel[0] = root_lin_vel[1] if num_frames > 1 else 0

    # Compute root angular velocity from quaternions
    root_ang_vel = quaternion_to_angular_velocity(root_rot, dt)

    # Extract root height
    root_height = root_pos[:, 2:3]  # z-coordinate

    # Estimate foot contacts
    foot_contacts = estimate_foot_contacts(dof_pos)

    # Build 29-dim AMP features
    # [dof_pos(9), dof_vel(9), root_lin_vel(3), root_ang_vel(3), root_height(1), foot_contacts(4)]
    features = np.concatenate([
        dof_pos,           # 0-8: joint positions
        dof_vel,           # 9-17: joint velocities
        root_lin_vel,      # 18-20: root linear velocity
        root_ang_vel,      # 21-23: root angular velocity
        root_height,       # 24: root height
        foot_contacts,     # 25-28: foot contacts
    ], axis=1).astype(np.float32)

    return {
        # AMP features
        'features': features,
        'feature_dim': 29,

        # Raw data (for reference/debugging)
        'dof_pos': dof_pos.astype(np.float32),
        'dof_vel': dof_vel.astype(np.float32),
        'root_pos': root_pos.astype(np.float32),
        'root_rot': root_rot.astype(np.float32),
        'root_lin_vel': root_lin_vel.astype(np.float32),
        'root_ang_vel': root_ang_vel.astype(np.float32),
        'foot_contacts': foot_contacts,

        # Metadata
        'fps': target_fps,
        'dt': dt,
        'num_frames': num_frames,
        'duration_sec': motion_data['duration_sec'],
        'source_file': motion_data.get('source_file', 'unknown'),
        'robot': motion_data.get('robot', 'wildrobot'),

        # Joint names from robot_config.yaml (no hardcoding)
        'joint_names': get_joint_names(),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert GMR output to AMP format")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input GMR pickle file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output AMP pickle file",
    )
    parser.add_argument(
        "--robot-config",
        type=str,
        required=True,
        help="Path to robot_config.yaml (required for joint order)",
    )
    parser.add_argument(
        "--target_fps",
        type=float,
        default=50.0,
        help="Target FPS for AMP training",
    )
    parser.add_argument(
        "--save_features_npy",
        action="store_true",
        help="Also save features as .npy file",
    )

    args = parser.parse_args()

    # Load robot config first (required for joint order)
    print(f"[bold blue]Loading robot config:[/bold blue] {args.robot_config}")
    config = load_robot_config(args.robot_config)
    joint_names = config['actuators']['joints']
    print(f"  Joint order: {joint_names}")

    # Load input
    print(f"[bold blue]Loading:[/bold blue] {args.input}")
    with open(args.input, 'rb') as f:
        motion_data = pickle.load(f)

    print(f"  Source FPS: {motion_data['fps']:.1f}")
    print(f"  Frames: {motion_data['num_frames']}")
    print(f"  Duration: {motion_data['duration_sec']:.2f}s")

    # Convert to AMP format
    print(f"\n[bold blue]Converting to AMP format...[/bold blue]")
    amp_data = convert_to_amp_format(motion_data, target_fps=args.target_fps)

    print(f"  Target FPS: {amp_data['fps']}")
    print(f"  Output frames: {amp_data['num_frames']}")
    print(f"  Feature dim: {amp_data['feature_dim']}")
    print(f"  Features shape: {amp_data['features'].shape}")

    # Save output
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(amp_data, f)
    print(f"\n[bold green]Saved to:[/bold green] {output_path}")

    # Optionally save features as numpy
    if args.save_features_npy:
        npy_path = str(output_path).replace('.pkl', '_features.npy')
        np.save(npy_path, amp_data['features'])
        print(f"[green]Features saved to:[/green] {npy_path}")

    # Print feature statistics
    features = amp_data['features']
    print(f"\n[bold blue]Feature Statistics:[/bold blue]")
    print(f"  Min: {features.min():.4f}")
    print(f"  Max: {features.max():.4f}")
    print(f"  Mean: {features.mean():.4f}")
    print(f"  Std: {features.std():.4f}")

    # Per-component statistics
    labels = [
        "Joint pos (0-8)", "Joint vel (9-17)", "Root lin vel (18-20)",
        "Root ang vel (21-23)", "Root height (24)", "Foot contacts (25-28)"
    ]
    ranges = [(0, 9), (9, 18), (18, 21), (21, 24), (24, 25), (25, 29)]

    print(f"\n[bold blue]Per-Component Statistics:[/bold blue]")
    for label, (start, end) in zip(labels, ranges):
        component = features[:, start:end]
        print(f"  {label}: min={component.min():.3f}, max={component.max():.3f}, mean={component.mean():.3f}")


if __name__ == "__main__":
    main()
