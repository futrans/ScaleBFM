import numpy as np
from scipy.spatial.transform import Rotation as R


def _slerp_quaternions(quaternions, lower, upper, alpha):
    q0 = quaternions[lower]
    q1 = quaternions[upper]

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)

    alpha_shape = (len(alpha),) + (1,) * (q0.ndim - 1)
    t = alpha.reshape(alpha_shape)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    near = sin_theta < 1e-7
    safe_sin = np.where(near, 1.0, sin_theta)
    interpolated = (
        np.sin((1.0 - t) * theta) / safe_sin * q0
        + np.sin(t * theta) / safe_sin * q1
    )
    linear = (1.0 - t) * q0 + t * q1
    interpolated = np.where(near, linear, interpolated)
    interpolated /= np.linalg.norm(interpolated, axis=-1, keepdims=True)

    return interpolated


def _slerp_rotvec(rotvec, lower, upper, alpha):
    shape = rotvec.shape
    quaternions = R.from_rotvec(rotvec.reshape(-1, 3)).as_quat().reshape(
        *shape[:-1], 4
    )
    interpolated = _slerp_quaternions(quaternions, lower, upper, alpha)
    return R.from_quat(interpolated.reshape(-1, 4)).as_rotvec().reshape(
        len(alpha), *shape[1:]
    )


def _resampling_indices(num_frames, source_fps, target_fps):
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    if num_frames < 2 or np.isclose(source_fps, target_fps):
        return None, source_fps

    duration = (num_frames - 1) / source_fps
    new_num_frames = max(2, round(duration * target_fps) + 1)
    target_time = np.linspace(0, num_frames - 1, new_num_frames)
    lower = np.floor(target_time).astype(np.int64)
    upper = np.minimum(lower + 1, num_frames - 1)
    alpha = target_time - lower
    actual_fps = (new_num_frames - 1) / duration
    return (lower, upper, alpha), actual_fps


def resample_bvh_motion(bvh_data, target_fps=30):
    indices, actual_fps = _resampling_indices(
        len(bvh_data.pos), bvh_data.fps, target_fps
    )
    if indices is None:
        return actual_fps

    lower, upper, alpha = indices
    position_alpha = alpha.reshape((len(alpha),) + (1,) * (bvh_data.pos.ndim - 1))
    bvh_data.pos = (
        (1.0 - position_alpha) * bvh_data.pos[lower]
        + position_alpha * bvh_data.pos[upper]
    )
    bvh_data.quats = _slerp_quaternions(
        bvh_data.quats, lower, upper, alpha
    )
    return actual_fps


def resample_smplx_motion(
    global_orient,
    full_body_pose,
    joints,
    source_fps,
    target_fps=30,
):
    num_frames = len(global_orient)
    indices, actual_fps = _resampling_indices(
        num_frames, source_fps, target_fps
    )
    if indices is None:
        return global_orient, full_body_pose, joints, source_fps

    lower, upper, alpha = indices
    new_num_frames = len(alpha)

    global_orient = _slerp_rotvec(global_orient, lower, upper, alpha)
    full_body_pose = _slerp_rotvec(full_body_pose, lower, upper, alpha)

    flat_joints = joints.reshape(num_frames, -1)
    joints = (
        (1.0 - alpha[:, None]) * flat_joints[lower]
        + alpha[:, None] * flat_joints[upper]
    ).reshape(new_num_frames, *joints.shape[1:])

    return global_orient, full_body_pose, joints, actual_fps
