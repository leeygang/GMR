#!/usr/bin/env python3
"""Convert GMR retargeted motion to AMP training format.

Converts the pickle output from GMR (smplx_to_robot_headless.py) to the format
needed for AMP discriminator training in WildRobot.

v0.7.0: All velocities are now in HEADING-LOCAL frame (Contract A).
This provides rotation invariance for the AMP discriminator.

v0.9.3: FK-based contact estimation using MuJoCo forward kinematics.
This fixes the "84% double stance" issue caused by the hip-pitch heuristic.

The AMP format includes:
- Joint positions (N DOF from robot_config)
- Joint velocities (N DOF) - computed from positions
- Root linear velocity (3D) - heading-local frame
- Root angular velocity (3D) - heading-local frame
- Root height (1D) - waist height
- Foot contact estimates (4D - left/right toe/heel)

Total: 2*N + 11 features per frame (e.g., 27 for 8-joint robot, 29 for 9-joint)

Joint order and count are loaded from robot_config.yaml (no hardcoding).

Usage:
    # Basic (uses hip-pitch heuristic for contacts - legacy)
    uv run python scripts/convert_to_amp_format.py \
        --input /path/to/gmr_output.pkl \
        --output /path/to/amp_motion.pkl \
        --robot-config /path/to/robot_config.yaml \
        --target_fps 50

    # With FK-based contacts (recommended - v0.9.3)
    uv run python scripts/convert_to_amp_format.py \
        --input /path/to/gmr_output.pkl \
        --output /path/to/amp_motion.pkl \
        --robot-config /path/to/robot_config.yaml \
        --robot-model /path/to/scene.xml \
        --target_fps 50
"""

import argparse
import pathlib
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from rich import print
from scipy.spatial.transform import Rotation as R

# Optional MuJoCo import for FK-based contact estimation
try:
    import mujoco

    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    mujoco = None


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
    with open(config_path, "r") as f:
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
    return config["actuators"]["joints"]


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


def quaternion_to_yaw(quats: np.ndarray) -> np.ndarray:
    """Extract yaw (heading) angle from quaternions.

    Args:
        quats: Quaternions in xyzw format, shape (N, 4)

    Returns:
        Yaw angles in radians, shape (N,)
    """
    # Convert xyzw to wxyz for standard formula
    x, y, z, w = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    # Extract yaw (rotation around z-axis)
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return yaw


def world_to_heading_local(
    vectors: np.ndarray,
    quats: np.ndarray,
) -> np.ndarray:
    """Convert world-frame vectors to heading-local frame (yaw-removed).

    The heading-local frame is the world frame rotated by negative yaw.
    This provides rotation invariance for the AMP discriminator.

    v0.7.0: Contract A - all velocities in heading-local frame.

    Args:
        vectors: 3D vectors in world frame, shape (N, 3)
        quats: Root quaternions in xyzw format, shape (N, 4)

    Returns:
        Vectors in heading-local frame, shape (N, 3)
    """
    # Extract yaw from quaternions
    yaw = quaternion_to_yaw(quats)

    # Compute sin/cos of negative yaw (to rotate back to heading-local)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # Apply R_z(-yaw) to each vector
    # R_z(-yaw) = [[cos_yaw, sin_yaw, 0], [-sin_yaw, cos_yaw, 0], [0, 0, 1]]
    vx, vy, vz = vectors[:, 0], vectors[:, 1], vectors[:, 2]

    vx_local = cos_yaw * vx + sin_yaw * vy
    vy_local = -sin_yaw * vx + cos_yaw * vy
    vz_local = vz  # z component unchanged

    return np.stack([vx_local, vy_local, vz_local], axis=1)


def quaternion_to_angular_velocity(
    quats: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Compute angular velocity from quaternion time series using vectorized SciPy.

    v0.7.2: Vectorized implementation for ~88x speedup over loop-based version.
    Uses SciPy Rotation (battle-tested, industry standard).
    Returns WORLD-FRAME angular velocity to match MuJoCo frameangvel.

    Method (vectorized):
    1. Batch create all Rotation objects at once
    2. Compute relative rotations: R_delta = R_curr * R_prev.inv() (batch)
    3. Convert to rotation vectors (batch)
    4. Angular velocity: ω = rotvec / dt

    For body→world convention, R_delta = R_curr * R_prev.inv() gives
    a world-frame rotation increment, so ω is in world frame.

    Args:
        quats: Quaternions in xyzw format (SciPy convention), shape (N, 4)
        dt: Time step

    Returns:
        Angular velocities in WORLD frame, shape (N, 3)
    """
    N = len(quats)
    if N < 2:
        return np.zeros((N, 3), dtype=np.float32)

    # Batch create all Rotation objects at once (vectorized)
    rotations = R.from_quat(quats)

    # Relative rotations: R_delta = R_curr * R_prev.inv() for all frames
    # This gives world-frame rotation increments
    r_prev = rotations[:-1]  # frames 0 to N-2
    r_curr = rotations[1:]  # frames 1 to N-1
    r_delta = r_curr * r_prev.inv()

    # Convert to rotation vectors (batch) and compute angular velocity
    omega = r_delta.as_rotvec() / dt

    # Prepend first frame (copy from second frame)
    omega = np.vstack([omega[0:1], omega])

    return omega.astype(np.float32)


def resample_motion(motion_data: Dict[str, Any], target_fps: float) -> Dict[str, Any]:
    """Resample motion to target FPS using linear interpolation.

    Args:
        motion_data: Original motion data
        target_fps: Target frames per second

    Returns:
        Resampled motion data
    """
    source_fps = motion_data["fps"]
    num_frames = motion_data["num_frames"]
    duration = motion_data["duration_sec"]

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
        "fps": target_fps,
        "root_pos": interp_array(motion_data["root_pos"]),
        "root_rot": interp_quat(motion_data["root_rot"]),
        "dof_pos": interp_array(motion_data["dof_pos"]),
        "source_file": motion_data.get("source_file", "unknown"),
        "robot": motion_data.get("robot", "wildrobot"),
        "num_frames": new_num_frames,
        "duration_sec": duration,
    }


def estimate_foot_contacts(
    dof_pos: np.ndarray,
    threshold_angle: float = 0.1,
) -> np.ndarray:
    """Estimate foot contacts from joint positions (LEGACY - hip pitch heuristic).

    Uses hip pitch angles to estimate gait phase.
    When hip pitch is negative (leg behind), foot is likely in contact.

    WARNING: This heuristic often produces 80%+ double stance due to
    the permissive threshold. Use estimate_foot_contacts_fk() instead.

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
    idx_left_hip_pitch = get_joint_idx("left_hip_pitch")
    idx_left_knee = get_joint_idx("left_knee_pitch")
    idx_right_hip_pitch = get_joint_idx("right_hip_pitch")
    idx_right_knee = get_joint_idx("right_knee_pitch")

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


def estimate_foot_contacts_fk(
    dof_pos: np.ndarray,
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    mj_model: "mujoco.MjModel",
    contact_height_threshold: float = 0.02,
    left_foot_geom_name: str = "left_toe",
    right_foot_geom_name: str = "right_toe",
) -> np.ndarray:
    """Estimate foot contacts using MuJoCo forward kinematics (v0.9.3).

    This method computes actual foot positions using FK and determines
    contact based on foot height above ground. Much more accurate than
    the hip-pitch heuristic.

    For each frame:
    1. Set MuJoCo qpos from root pose + joint angles
    2. Call mj_forward() to compute forward kinematics
    3. Read foot geom z-positions
    4. Contact = foot.z < threshold

    Args:
        dof_pos: Joint positions (N, num_joints)
        root_pos: Root positions (N, 3)
        root_rot: Root quaternions in xyzw format (N, 4)
        mj_model: MuJoCo model with foot geoms
        contact_height_threshold: Height below which foot is in contact (m)
        left_foot_geom_name: Name of left foot geom in model
        right_foot_geom_name: Name of right foot geom in model

    Returns:
        Foot contacts (N, 4) - [left_toe, left_heel, right_toe, right_heel]
    """
    if not MUJOCO_AVAILABLE:
        raise RuntimeError(
            "MuJoCo is required for FK-based contact estimation. "
            "Install with: pip install mujoco"
        )

    N = len(dof_pos)
    contacts = np.zeros((N, 4), dtype=np.float32)

    # Create data object for forward kinematics
    mj_data = mujoco.MjData(mj_model)

    # Get foot geom IDs
    try:
        left_foot_geom_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_GEOM, left_foot_geom_name
        )
        right_foot_geom_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_GEOM, right_foot_geom_name
        )
    except Exception as e:
        raise ValueError(
            f"Could not find foot geoms '{left_foot_geom_name}' and/or "
            f"'{right_foot_geom_name}' in model. Error: {e}"
        )

    # Track contact statistics for reporting
    left_contact_frames = 0
    right_contact_frames = 0
    double_stance_frames = 0

    for i in range(N):
        # Set root position
        mj_data.qpos[0:3] = root_pos[i]

        # Set root orientation (convert xyzw to wxyz for MuJoCo)
        x, y, z, w = root_rot[i]
        mj_data.qpos[3:7] = [w, x, y, z]

        # Set joint positions
        num_joints = dof_pos.shape[1]
        mj_data.qpos[7 : 7 + num_joints] = dof_pos[i]

        # Run forward kinematics
        mujoco.mj_forward(mj_model, mj_data)

        # Read foot heights from geom positions
        left_foot_z = mj_data.geom_xpos[left_foot_geom_id, 2]
        right_foot_z = mj_data.geom_xpos[right_foot_geom_id, 2]

        # Determine contact based on height threshold
        left_contact = 1.0 if left_foot_z < contact_height_threshold else 0.0
        right_contact = 1.0 if right_foot_z < contact_height_threshold else 0.0

        # Track statistics
        if left_contact > 0.5:
            left_contact_frames += 1
        if right_contact > 0.5:
            right_contact_frames += 1
        if left_contact > 0.5 and right_contact > 0.5:
            double_stance_frames += 1

        # Set toe and heel contacts (same value for each foot)
        contacts[i, 0] = left_contact  # left toe
        contacts[i, 1] = left_contact  # left heel
        contacts[i, 2] = right_contact  # right toe
        contacts[i, 3] = right_contact  # right heel

    # Report contact statistics
    left_pct = 100.0 * left_contact_frames / N
    right_pct = 100.0 * right_contact_frames / N
    double_pct = 100.0 * double_stance_frames / N

    print(f"  [cyan]FK Contact Statistics:[/cyan]")
    print(f"    Left foot contact: {left_contact_frames}/{N} frames ({left_pct:.1f}%)")
    print(
        f"    Right foot contact: {right_contact_frames}/{N} frames ({right_pct:.1f}%)"
    )
    print(f"    Double stance: {double_stance_frames}/{N} frames ({double_pct:.1f}%)")

    return contacts


def load_mujoco_model(model_path: str) -> Optional["mujoco.MjModel"]:
    """Load MuJoCo model from XML file.

    Handles nested includes by loading all assets from the model directory.
    XML files are loaded as bytes (MuJoCo expects this).

    Args:
        model_path: Path to MuJoCo XML file

    Returns:
        MuJoCo model or None if loading fails
    """
    if not MUJOCO_AVAILABLE:
        print("[yellow]Warning: MuJoCo not available, cannot load model[/yellow]")
        return None

    model_path = pathlib.Path(model_path)
    if not model_path.exists():
        print(f"[yellow]Warning: Model file not found: {model_path}[/yellow]")
        return None

    try:
        # Build assets dict - all files loaded as bytes (MuJoCo convention)
        assets_dir = model_path.parent
        assets = {}

        # Load all XML files as bytes
        for xml_file in assets_dir.glob("*.xml"):
            try:
                assets[xml_file.name] = xml_file.read_bytes()
            except Exception:
                pass

        # Load binary assets (meshes, textures)
        for asset_file in assets_dir.glob("*"):
            if asset_file.is_file() and asset_file.suffix in [
                ".stl",
                ".obj",
                ".png",
                ".jpg",
            ]:
                try:
                    assets[asset_file.name] = asset_file.read_bytes()
                except Exception:
                    pass

        # Also check assets subdirectory for meshes
        meshes_dir = assets_dir / "assets"
        if meshes_dir.exists():
            for mesh_file in meshes_dir.glob("*.stl"):
                try:
                    assets[mesh_file.name] = mesh_file.read_bytes()
                except Exception:
                    pass

        # Load model
        xml_content = model_path.read_text()
        mj_model = mujoco.MjModel.from_xml_string(xml_content, assets)
        return mj_model

    except Exception as e:
        print(f"[yellow]Warning: Failed to load MuJoCo model: {e}[/yellow]")
        return None


def convert_to_amp_format(
    motion_data: Dict[str, Any],
    target_fps: float = 50.0,
    mj_model: Optional["mujoco.MjModel"] = None,
    contact_height_threshold: float = 0.02,
) -> Dict[str, Any]:
    """Convert GMR output to AMP training format.

    v0.9.3: If mj_model is provided, uses FK-based contact estimation
    (recommended). Otherwise falls back to hip-pitch heuristic (legacy).

    Args:
        motion_data: Output from GMR retargeting
        target_fps: Target FPS for AMP training
        mj_model: Optional MuJoCo model for FK-based contact estimation
        contact_height_threshold: Foot height threshold for contact (m)

    Returns:
        AMP-formatted motion data
    """
    # Resample to target FPS if needed
    source_fps = motion_data["fps"]
    if abs(source_fps - target_fps) > 0.1:
        print(
            f"[yellow]Resampling from {source_fps:.1f} FPS to {target_fps:.1f} FPS[/yellow]"
        )
        motion_data = resample_motion(motion_data, target_fps)

    dt = 1.0 / target_fps
    num_frames = motion_data["num_frames"]

    dof_pos = motion_data["dof_pos"]
    root_pos = motion_data["root_pos"]
    root_rot = motion_data["root_rot"]  # xyzw format

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

    # =========================================================
    # v0.7.0: Convert velocities to heading-local frame (Contract A)
    # This provides rotation invariance for the AMP discriminator
    # =========================================================
    print(
        "  [cyan]Converting velocities to heading-local frame (v0.7.0 Contract A)[/cyan]"
    )

    # Convert root linear velocity: world → heading-local
    root_lin_vel_heading = world_to_heading_local(root_lin_vel, root_rot)

    # Convert root angular velocity: world → heading-local
    root_ang_vel_heading = world_to_heading_local(root_ang_vel, root_rot)

    # Extract root height (waist height - consistent with robot root body)
    root_height = root_pos[:, 2:3]  # z-coordinate

    # =========================================================
    # v0.9.3: Foot contact estimation
    # Use FK-based method if MuJoCo model is available (recommended)
    # Otherwise fall back to hip-pitch heuristic (legacy)
    # =========================================================
    if mj_model is not None:
        print("  [green]Using FK-based contact estimation (v0.9.3)[/green]")
        foot_contacts = estimate_foot_contacts_fk(
            dof_pos,
            root_pos,
            root_rot,
            mj_model,
            contact_height_threshold=contact_height_threshold,
        )
        contact_method = "fk"
    else:
        print("  [yellow]Using hip-pitch heuristic for contacts (legacy)[/yellow]")
        print(
            "  [yellow]Tip: Pass --robot-model for more accurate FK-based contacts[/yellow]"
        )
        foot_contacts = estimate_foot_contacts(dof_pos)
        contact_method = "heuristic"

    # Build AMP features dynamically based on robot DOF
    # [dof_pos(N), dof_vel(N), root_lin_vel(3), root_ang_vel(3), root_height(1), foot_contacts(4)]
    # v0.7.0: All velocities are in heading-local frame
    num_joints = dof_pos.shape[1]
    feature_dim = num_joints * 2 + 11  # 2*N + 11
    features = np.concatenate(
        [
            dof_pos,  # 0-8: joint positions
            dof_vel,  # 9-17: joint velocities
            root_lin_vel_heading,  # 18-20: root linear velocity (heading-local)
            root_ang_vel_heading,  # 21-23: root angular velocity (heading-local)
            root_height,  # 24: root height (waist)
            foot_contacts,  # 25-28: foot contacts
        ],
        axis=1,
    ).astype(np.float32)

    return {
        # AMP features
        "features": features,
        "feature_dim": feature_dim,
        # Raw data (for reference/debugging)
        "dof_pos": dof_pos.astype(np.float32),
        "dof_vel": dof_vel.astype(np.float32),
        "root_pos": root_pos.astype(np.float32),
        "root_rot": root_rot.astype(np.float32),
        "root_lin_vel": root_lin_vel.astype(np.float32),
        "root_ang_vel": root_ang_vel.astype(np.float32),
        "foot_contacts": foot_contacts,
        # Metadata
        "fps": target_fps,
        "dt": dt,
        "num_frames": num_frames,
        "duration_sec": motion_data["duration_sec"],
        "source_file": motion_data.get("source_file", "unknown"),
        "robot": motion_data.get("robot", "wildrobot"),
        # Joint names from robot_config.yaml (no hardcoding)
        "joint_names": get_joint_names(),
        # v0.9.3: Track which contact method was used
        "contact_method": contact_method,
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

    # Load input
    print(f"[bold blue]Loading:[/bold blue] {args.input}")
    with open(args.input, "rb") as f:
        motion_data = pickle.load(f)

    print(f"  Source FPS: {motion_data['fps']:.1f}")
    print(f"  Frames: {motion_data['num_frames']}")
    print(f"  Duration: {motion_data['duration_sec']:.2f}s")

    # Convert to AMP format
    print(f"\n[bold blue]Converting to AMP format...[/bold blue]")
    amp_data = convert_to_amp_format(
        motion_data,
        target_fps=args.target_fps,
        mj_model=mj_model,
        contact_height_threshold=args.contact_threshold,
    )

    print(f"  Target FPS: {amp_data['fps']}")
    print(f"  Output frames: {amp_data['num_frames']}")
    print(f"  Feature dim: {amp_data['feature_dim']}")
    print(f"  Features shape: {amp_data['features'].shape}")
    print(f"  Contact method: {amp_data['contact_method']}")

    # Save output
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(amp_data, f)
    print(f"\n[bold green]Saved to:[/bold green] {output_path}")

    # Optionally save features as numpy
    if args.save_features_npy:
        npy_path = str(output_path).replace(".pkl", "_features.npy")
        np.save(npy_path, amp_data["features"])
        print(f"[green]Features saved to:[/green] {npy_path}")

    # Print feature statistics
    features = amp_data["features"]
    print(f"\n[bold blue]Feature Statistics:[/bold blue]")
    print(f"  Min: {features.min():.4f}")
    print(f"  Max: {features.max():.4f}")
    print(f"  Mean: {features.mean():.4f}")
    print(f"  Std: {features.std():.4f}")

    # Per-component statistics (dynamic based on feature_dim)
    num_joints = len(config["actuators"]["joints"])
    labels = [
        f"Joint pos (0-{num_joints-1})",
        f"Joint vel ({num_joints}-{2*num_joints-1})",
        f"Root lin vel ({2*num_joints}-{2*num_joints+2})",
        f"Root ang vel ({2*num_joints+3}-{2*num_joints+5})",
        f"Root height ({2*num_joints+6})",
        f"Foot contacts ({2*num_joints+7}-{2*num_joints+10})",
    ]
    ranges = [
        (0, num_joints),
        (num_joints, 2 * num_joints),
        (2 * num_joints, 2 * num_joints + 3),
        (2 * num_joints + 3, 2 * num_joints + 6),
        (2 * num_joints + 6, 2 * num_joints + 7),
        (2 * num_joints + 7, 2 * num_joints + 11),
    ]

    print(f"\n[bold blue]Per-Component Statistics:[/bold blue]")
    for label, (start, end) in zip(labels, ranges):
        component = features[:, start:end]
        print(
            f"  {label}: min={component.min():.3f}, max={component.max():.3f}, mean={component.mean():.3f}"
        )


if __name__ == "__main__":
    main()
