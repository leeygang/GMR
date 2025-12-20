# WildRobot Motion Retargeting Guidance

This document describes how to regenerate the IK configuration and motion files when `wildrobot.xml` is modified.

## Prerequisites

### macOS
- GMR project: `/Users/ygli/projects/GMR`
- WildRobot project: `/Users/ygli/projects/wildrobot`
- AMASS dataset: `/Users/ygli/projects/amass/smplx`
- Python environment: `uv` (runs via `uv run python`)

### Ubuntu
- GMR project: `~/projects/GMR`
- WildRobot project: `~/projects/wildrobot`
- AMASS dataset: `~/projects/amass/smplx`
- Python environment: virtualenv or conda (runs via `python` after activation)

```bash
# Ubuntu setup (one-time)
cd ~/projects/GMR
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 1. Regenerate IK Config

When `wildrobot.xml` joint indices, ranges, or body hierarchy changes, update the IK config:

### 1.1 Check Joint Configuration

First, inspect the current joint configuration in `wildrobot.xml`:

```bash
# View joint definitions
grep -A2 "<joint" /Users/ygli/projects/wildrobot/assets/wildrobot.xml
```

Expected joint order (DoF 7+):
| Index | Joint Name | Range |
|-------|------------|-------|
| 7 | left_hip_pitch | -5° to 90° |
| 8 | left_hip_roll | -90° to 10° |
| 9 | left_knee_pitch | 0° to 80° |
| 10 | left_ankle_pitch | -45° to 45° |
| 11 | right_hip_pitch | -90° to 5° |
| 12 | right_hip_roll | -10° to 90° |
| 13 | right_knee_pitch | 0° to 80° |
| 14 | right_ankle_pitch | -45° to 45° |

### 1.2 Update IK Config

Edit the IK config file:
```
/Users/ygli/projects/GMR/general_motion_retargeting/ik_configs/smplx_to_wildrobot.json
```

Key parameters to adjust:
- `human_scale_table`: Scale factor for human-to-robot mapping (currently 0.55)
- `ik_match_table1/2`: Body mapping with position weights and rotation offsets
- Rotation offset quaternion `[0.5, -0.5, -0.5, -0.5]` works for G1-style robots

### 1.3 Verify Robot Path

Ensure `params.py` points to the correct XML:

```python
# In /Users/ygli/projects/GMR/general_motion_retargeting/params.py
"wildrobot": pathlib.Path("/Users/ygli/projects/wildrobot/assets/scene_flat_terrain.xml"),
```

## 2. Regenerate Motion Files

### 2.1 Retarget a Single Motion

#### macOS (using uv)

```bash
cd /Users/ygli/projects/GMR

# Walking motion
uv run python scripts/smplx_to_robot_headless.py \
    --smplx_file /Users/ygli/projects/amass/smplx/KIT/3/walking_medium10_stageii.npz \
    --robot wildrobot \
    --save_path /Users/ygli/projects/wildrobot/assets/motions/walking_medium10.pkl

# Running motion
uv run python scripts/smplx_to_robot_headless.py \
    --smplx_file /Users/ygli/projects/amass/smplx/KIT/359/walking_run04_stageii.npz \
    --robot wildrobot \
    --save_path /Users/ygli/projects/wildrobot/assets/motions/walking_run04.pkl

# Turning motion
uv run python scripts/smplx_to_robot_headless.py \
    --smplx_file /Users/ygli/projects/amass/smplx/KIT/167/turn_left05_stageii.npz \
    --robot wildrobot \
    --save_path /Users/ygli/projects/wildrobot/assets/motions/turn_left05.pkl
```

#### Ubuntu (using virtualenv)

```bash
cd ~/projects/GMR
source .venv/bin/activate

# Walking motion
python scripts/smplx_to_robot_headless.py \
    --smplx_file ~/projects/amass/smplx/KIT/3/walking_medium10_stageii.npz \
    --robot wildrobot \
    --save_path ~/projects/wildrobot/assets/motions/walking_medium10.pkl

# Running motion
python scripts/smplx_to_robot_headless.py \
    --smplx_file ~/projects/amass/smplx/KIT/359/walking_run04_stageii.npz \
    --robot wildrobot \
    --save_path ~/projects/wildrobot/assets/motions/walking_run04.pkl

# Turning motion
python scripts/smplx_to_robot_headless.py \
    --smplx_file ~/projects/amass/smplx/KIT/167/turn_left05_stageii.npz \
    --robot wildrobot \
    --save_path ~/projects/wildrobot/assets/motions/turn_left05.pkl
```

### 2.2 Generate Videos

#### macOS

```bash
cd /Users/ygli/projects/GMR

uv run python scripts/render_robot_motion.py \
    --robot wildrobot \
    --motion_path /Users/ygli/projects/wildrobot/assets/motions/walking_medium10.pkl \
    --output_video /Users/ygli/projects/wildrobot/assets/motions/walking_medium10.mp4
```

#### Ubuntu

```bash
cd ~/projects/GMR
source .venv/bin/activate

python scripts/render_robot_motion.py \
    --robot wildrobot \
    --motion_path ~/projects/wildrobot/assets/motions/walking_medium10.pkl \
    --output_video ~/projects/wildrobot/assets/motions/walking_medium10.mp4
```

## 3. Test and Verify

### 3.1 Quick Knee Bending Check

Run this script to verify knee bending is within valid range:

#### macOS

```bash
cd /Users/ygli/projects/GMR && uv run python -c "
import pickle
import numpy as np

motions = [
    ('walking_medium10', '/Users/ygli/projects/wildrobot/assets/motions/walking_medium10.pkl'),
    ('walking_run04', '/Users/ygli/projects/wildrobot/assets/motions/walking_run04.pkl'),
    ('turn_left05', '/Users/ygli/projects/wildrobot/assets/motions/turn_left05.pkl'),
]

# DOF indices in dof_pos array:
# [0] waist_yaw, [1] left_hip_pitch, [2] left_hip_roll
# [3] left_knee_pitch, [4] left_ankle_pitch
# [5] right_hip_pitch, [6] right_hip_roll
# [7] right_knee_pitch, [8] right_ankle_pitch

print('=== Knee Bending Verification ===')
print('Valid range: 0° to 80°')
print()

all_pass = True
for name, path in motions:
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        dof_pos = data['dof_pos']
        left_knee = np.degrees(dof_pos[:, 3])
        right_knee = np.degrees(dof_pos[:, 7])
        
        left_ok = left_knee.min() >= -1 and left_knee.max() <= 81
        right_ok = right_knee.min() >= -1 and right_knee.max() <= 81
        
        status = '✅' if (left_ok and right_ok) else '❌'
        if not (left_ok and right_ok):
            all_pass = False
        
        print(f'{status} {name}')
        print(f'   Left knee:  {left_knee.min():.1f}° to {left_knee.max():.1f}°')
        print(f'   Right knee: {right_knee.min():.1f}° to {right_knee.max():.1f}°')
    except FileNotFoundError:
        print(f'⚠️  {name}: File not found')
        all_pass = False

print()
print('Overall:', '✅ PASS' if all_pass else '❌ FAIL')
"
```

#### Ubuntu

```bash
cd ~/projects/GMR && source .venv/bin/activate && python -c "
import pickle
import numpy as np

motions = [
    ('walking_medium10', '$HOME/projects/wildrobot/assets/motions/walking_medium10.pkl'),
    ('walking_run04', '$HOME/projects/wildrobot/assets/motions/walking_run04.pkl'),
    ('turn_left05', '$HOME/projects/wildrobot/assets/motions/turn_left05.pkl'),
]

import os
motions = [(n, os.path.expandvars(p)) for n, p in motions]

print('=== Knee Bending Verification ===')
print('Valid range: 0° to 80°')
print()

all_pass = True
for name, path in motions:
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        dof_pos = data['dof_pos']
        left_knee = np.degrees(dof_pos[:, 3])
        right_knee = np.degrees(dof_pos[:, 7])
        
        left_ok = left_knee.min() >= -1 and left_knee.max() <= 81
        right_ok = right_knee.min() >= -1 and right_knee.max() <= 81
        
        status = '✅' if (left_ok and right_ok) else '❌'
        if not (left_ok and right_ok):
            all_pass = False
        
        print(f'{status} {name}')
        print(f'   Left knee:  {left_knee.min():.1f}° to {left_knee.max():.1f}°')
        print(f'   Right knee: {right_knee.min():.1f}° to {right_knee.max():.1f}°')
    except FileNotFoundError:
        print(f'⚠️  {name}: File not found')
        all_pass = False

print()
print('Overall:', '✅ PASS' if all_pass else '❌ FAIL')
"
```

### 3.2 Full Joint Range Check

Verify all joints stay within their limits:

#### macOS

```bash
cd /Users/ygli/projects/GMR && uv run python -c "
import pickle
import numpy as np

# Joint limits from wildrobot.xml (in degrees)
JOINT_LIMITS = {
    0: ('waist_yaw', -45, 45),
    1: ('left_hip_pitch', -5, 90),
    2: ('left_hip_roll', -90, 10),
    3: ('left_knee_pitch', 0, 80),
    4: ('left_ankle_pitch', -45, 45),
    5: ('right_hip_pitch', -90, 5),
    6: ('right_hip_roll', -10, 90),
    7: ('right_knee_pitch', 0, 80),
    8: ('right_ankle_pitch', -45, 45),
}

motion_path = '/Users/ygli/projects/wildrobot/assets/motions/walking_medium10.pkl'

with open(motion_path, 'rb') as f:
    data = pickle.load(f)

dof_pos = np.degrees(data['dof_pos'])

print('=== Joint Limit Verification ===')
print(f'Motion: {motion_path}')
print(f'Frames: {len(dof_pos)}')
print()

all_pass = True
for idx, (name, low, high) in JOINT_LIMITS.items():
    joint_vals = dof_pos[:, idx]
    min_val, max_val = joint_vals.min(), joint_vals.max()
    
    # Allow 1 degree tolerance
    in_range = min_val >= (low - 1) and max_val <= (high + 1)
    status = '✅' if in_range else '❌'
    if not in_range:
        all_pass = False
    
    print(f'{status} {name:20s}: {min_val:6.1f}° to {max_val:6.1f}° (limit: {low}° to {high}°)')

print()
print('Overall:', '✅ PASS' if all_pass else '❌ FAIL')
"
```

#### Ubuntu

```bash
cd ~/projects/GMR && source .venv/bin/activate && python -c "
import pickle
import numpy as np
import os

# Joint limits from wildrobot.xml (in degrees)
JOINT_LIMITS = {
    0: ('waist_yaw', -45, 45),
    1: ('left_hip_pitch', -5, 90),
    2: ('left_hip_roll', -90, 10),
    3: ('left_knee_pitch', 0, 80),
    4: ('left_ankle_pitch', -45, 45),
    5: ('right_hip_pitch', -90, 5),
    6: ('right_hip_roll', -10, 90),
    7: ('right_knee_pitch', 0, 80),
    8: ('right_ankle_pitch', -45, 45),
}

motion_path = os.path.expanduser('~/projects/wildrobot/assets/motions/walking_medium10.pkl')

with open(motion_path, 'rb') as f:
    data = pickle.load(f)

dof_pos = np.degrees(data['dof_pos'])

print('=== Joint Limit Verification ===')
print(f'Motion: {motion_path}')
print(f'Frames: {len(dof_pos)}')
print()

all_pass = True
for idx, (name, low, high) in JOINT_LIMITS.items():
    joint_vals = dof_pos[:, idx]
    min_val, max_val = joint_vals.min(), joint_vals.max()
    
    # Allow 1 degree tolerance
    in_range = min_val >= (low - 1) and max_val <= (high + 1)
    status = '✅' if in_range else '❌'
    if not in_range:
        all_pass = False
    
    print(f'{status} {name:20s}: {min_val:6.1f}° to {max_val:6.1f}° (limit: {low}° to {high}°)')

print()
print('Overall:', '✅ PASS' if all_pass else '❌ FAIL')
"
```

### 3.3 Visual Inspection

Watch the generated videos to verify:
1. Knees bend naturally during walking/running
2. No jittering or sudden joint flips
3. Feet don't penetrate the ground
4. Motion looks smooth and natural

## 4. Troubleshooting

### Knees Not Bending
- Check `human_scale_table` value (try 0.5-0.6 range)
- Verify knee joint limits in XML are correct (0° to 80°)
- Check rotation offsets in IK config

### Joint Limit Violations
- Adjust IK weights in `ik_match_table1/2`
- Lower position weights for problematic joints
- Verify XML joint ranges match IK config assumptions

### Motion Looks Wrong
- Check body name mapping in IK config matches XML body names
- Verify `robot_root_name` and `human_root_name` are correct
- Try different rotation offset quaternions

## 5. Reference Files

| File | Purpose |
|------|---------|
| `/Users/ygli/projects/wildrobot/assets/wildrobot.xml` | Robot URDF/MuJoCo definition |
| `/Users/ygli/projects/wildrobot/assets/scene_flat_terrain.xml` | Scene with robot |
| `/Users/ygli/projects/GMR/general_motion_retargeting/ik_configs/smplx_to_wildrobot.json` | IK configuration |
| `/Users/ygli/projects/GMR/general_motion_retargeting/params.py` | Robot path mapping |
| `/Users/ygli/projects/GMR/scripts/smplx_to_robot_headless.py` | Headless retargeting script |
| `/Users/ygli/projects/GMR/scripts/render_robot_motion.py` | Video rendering script |
