import numpy as np
from lerobot_robot_ufactory.devices.umi.vive_tracker.transformations import Transformations


POSE_KEYS = ('pose.x', 'pose.y', 'pose.z', 'pose.rx', 'pose.ry', 'pose.rz')

def _as_pose_array(pose) -> np.ndarray:
    arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    if arr.size < 6:
        raise ValueError(f"Expected at least 6 pose values, got shape {arr.shape}")
    return arr

def continuous_rotvec(new_rv, prev_rv):
    """Keep rotvec close to prev to avoid ±π flips.
    An axis-angle vector r = angle*axis has infinitely many equivalent
    representations: (angle + 2πk)*axis and (-angle + 2πk)*axis for any
    integer k. This picks the representation closest to prev_rv in vector
    space.
    """
    new_rv = np.asarray(new_rv, dtype=np.float64)
    prev_rv = np.asarray(prev_rv, dtype=np.float64)
    angle = np.linalg.norm(new_rv)
    if angle < 1e-12:
        return new_rv

    axis = new_rv / angle
    proj = np.dot(prev_rv, axis)

    # Two equivalent families on the same axis: +angle + 2πk and -angle + 2πk
    candidates = []
    for base in (angle, -angle):
        k = round((proj - base) / (2 * np.pi))
        candidates.append((base + 2 * np.pi * k) * axis)

    # Pick the candidate closest to prev_rv
    return min(candidates, key=lambda c: np.linalg.norm(c - prev_rv))

def blend_poses(pose_a, pose_b, alpha):
    """位姿混合: (1-alpha)*A + alpha*B, 旋转用 SO(3) 插值。
    先将 pose_b 的 rotvec 归一化到与 pose_a 同符号半球，避免 ±π 跳变破坏线性混合。"""
    blended_pos = (1 - alpha) * np.array(pose_a[:3]) + alpha * np.array(pose_b[:3])
    # 旋转用线性混合 rotvec (delta 很小时近似 SLERP)
    rot_b = continuous_rotvec(np.array(pose_b[3:6]), np.array(pose_a[3:6]))
    blended_rot = (1 - alpha) * np.array(pose_a[3:6]) + alpha * rot_b
    return np.concatenate([blended_pos, blended_rot]).tolist()

def compute_relative_axis_angle(rot_prev, rot_curr):
    """
    计算两个轴角之间的相对旋转。
    逻辑: R_diff = R_prev.T @ R_curr
    返回: 相对轴角向量
    """
    # 1. 转为矩阵
    R_prev = Transformations.rxryrz_to_rotation_matrix(*rot_prev)
    R_curr = Transformations.rxryrz_to_rotation_matrix(*rot_curr)
    
    # 2. 计算相对旋转矩阵
    # R_delta 表示从 prev 坐标系到 curr 坐标系的旋转
    R_delta = R_prev.T @ R_curr
    
    # 3. 转回轴角
    return Transformations.rotation_matrix_to_rxryrz(R_delta)

def compute_target_axis_angle(rot_prev, rot_delta):
    """
    根据起始轴角和相对轴角计算目标轴角
    """
    R_prev = Transformations.rxryrz_to_rotation_matrix(*rot_prev)
    R_delta = Transformations.rxryrz_to_rotation_matrix(*rot_delta)
    R_curr = R_prev @ R_delta
    # R_curr = R_prev.apply(R_delta)
    return Transformations.rotation_matrix_to_rxryrz(R_curr)

def to_relative_pose(prev, curr):
    prev = _as_pose_array(prev)
    curr = _as_pose_array(curr)
    R_prev = Transformations.rxryrz_to_rotation_matrix(*prev[3:6])
    R_curr = Transformations.rxryrz_to_rotation_matrix(*curr[3:6])

    # 纯 numpy: R.inv() = R.T, inv * v = R.T @ v, inv * R2 = R.T @ R2
    rel_pos = R_prev.T @ (curr[:3] - prev[:3])
    rel_rot = Transformations.rotation_matrix_to_rxryrz(R_prev.T @ R_curr)
    return np.concatenate([rel_pos, rel_rot]).astype(np.float64)

def to_absolute_pose(delta, prev):
    delta = _as_pose_array(delta)
    prev = _as_pose_array(prev)
    R_delta = Transformations.rxryrz_to_rotation_matrix(*delta[3:6])
    R_prev = Transformations.rxryrz_to_rotation_matrix(*prev[3:6])

    # 纯 numpy: apply(v) = R @ v, R1 * R2 = R1 @ R2
    abs_pos = prev[:3] + R_prev @ delta[:3]
    abs_rot = Transformations.rotation_matrix_to_rxryrz(R_prev @ R_delta)
    abs_rot = continuous_rotvec(abs_rot, prev[3:6])
    return np.concatenate([abs_pos, abs_rot]).astype(np.float64)

def get_pose_from_feature_dict(feature_dict, prefix=''):
    return np.array([feature_dict[f'{prefix}{key}'] for key in POSE_KEYS])

def update_feature_dict_from_pose(feature_dict, pose, prefix=''):
    for i, key in enumerate(POSE_KEYS):
        feature_dict[f'{prefix}{key}'] = pose[i]