# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import math
import os
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


def generate_points_in_sphere(n_points: int, radius: float, seed=None) -> torch.Tensor:
    """
    Uniformly sample points within a sphere of a specified radius.

    :param n_points: The number of points to generate.
    :param radius: The radius of the sphere.
    :return: A tensor of shape (n_points, 3), representing the (x, y, z) coordinates of the points.
    """
    seed = int(os.environ.get("SMI_GEOMETRY_SEED", "0")) if seed is None else int(seed)
    source = os.environ.get("SMI_GEOMETRY_POINTS")
    if source:
        points = torch.from_numpy(np.load(source, allow_pickle=False))
        if points.shape != (n_points, 3) or points.dtype != torch.float32:
            raise ValueError("Geometry fixture must be float32 with shape (n_points, 3)")
        if not torch.isfinite(points).all() or (points.norm(dim=1) > radius + 1e-5).any():
            raise ValueError("Geometry fixture contains invalid or out-of-sphere points")
    else:
        points = _sample_sphere(n_points, radius, seed)
    folder = os.environ.get("SMI_REPRO_DIR")
    if folder and int(os.environ.get("RANK", "0")) == 0:
        destination = Path(folder)
        destination.mkdir(parents=True, exist_ok=True)
        stem = f"geometry-{n_points}-{radius:g}"
        array = points.numpy()
        np.save(destination / f"{stem}.npy", array, allow_pickle=False)
        metadata = {"seed": seed, "radius": radius, "n_points": n_points,
                    "source": source, "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
        (destination / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return points


def _sample_sphere(n_points, radius, seed):
    # A separate CPU generator isolates FoV sampling from video/model RNG use.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples_r = torch.rand(n_points, generator=generator, dtype=torch.float32, device="cpu")
    samples_phi = torch.rand(n_points, generator=generator, dtype=torch.float32, device="cpu")
    samples_u = torch.rand(n_points, generator=generator, dtype=torch.float32, device="cpu")

    r = radius * torch.pow(samples_r, 1 / 3)
    phi = 2 * math.pi * samples_phi
    theta = torch.acos(1 - 2 * samples_u)

    # transfer the coordinates from spherical to cartesian
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    points = torch.stack((x, y, z), dim=1)
    return points


def rotation_matrix_to_angles(R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate the Pitch and Yaw angles from a 3x3 rotation matrix R in the camera coordinate system.

    Assumed Camera Coordinate System: X=Right, Y=Up, Z=Backward
    (or NeRF style: X=Right, Y=Down, Z=Forward).
    Here we adopt the common Computer Vision convention: Z-axis is Forward.

    Note: The angle calculations here are directly based on the conventions of your `is_inside_fov_3d_hv` function:
    - Yaw/Azimuth angle is in the XZ plane (atan2(x, z)).
    - Pitch/Elevation angle is relative to the horizontal plane (atan2(y, sqrt(x^2 + z^2))).

    For the third column R[:, 2] of the W2C matrix R (the direction of the World Z-axis in the Camera frame),
    this typically corresponds to the direction the camera is looking
    (the representation of the world Z-axis in the camera frame).

    To simplify and match your `is_inside_fov` logic, we directly use the camera's Z-axis vector:
    Camera Z-axis direction in World Frame (Forward Vector): fwd = R_w2c_inv @ [0, 0, 1]
    More simply, the Z-axis vector of the C2W matrix is the camera's forward vector in the world frame.
    C2W = W2C_inv
    """

    R_c2w = R.T
    fwd = R_c2w[:, 2]

    x = fwd[0]
    y = fwd[1]
    z = fwd[2]

    # compute yaw and pitch
    yaw_rad = torch.atan2(x, z)
    yaw_deg = yaw_rad * (180.0 / math.pi)
    pitch_rad = torch.atan2(y, torch.sqrt(x**2 + z**2))
    pitch_deg = pitch_rad * (180.0 / math.pi)

    return pitch_deg, yaw_deg


def is_inside_fov_3d_hv(
    points: torch.Tensor,
    center: torch.Tensor,
    center_pitch: torch.Tensor,
    center_yaw: torch.Tensor,
    fov_half_h: torch.Tensor,
    fov_half_v: torch.Tensor,
) -> torch.Tensor:
    """
    Check whether points are inside a 3D view frustum defined by a center coordinate, pitch angle, and yaw angle.

    :param points: Tensor of shape (N, 3) or (N, B, 3) representing the coordinates of the sampled points.
    :param center: Tensor of shape (3) or (B, 3) representing the camera center coordinates.
    :param center_pitch: Tensor of shape (1) or (B) representing the pitch angle of center view direction.
    :param center_yaw: Tensor of shape (1) or (B) representing the yaw angle of the center view direction.
    :param fov_half_h: The horizontal half field-of-view angle (in degrees).
    :param fov_half_v: The vertical half field-of-view angle (in degrees).
    :return: Boolean tensor of shape (N) or (N, B), indicating whether each point is inside the FOV.
    """
    if points.ndim == 2:  # N, 3
        vectors = points - center[None, :]
        C = 1
    elif points.ndim == 3:  # N, B, 3
        vectors = points - center[None, ...]
        center_pitch = center_pitch[None, :] if center_pitch.ndim == 1 else center_pitch
        center_yaw = center_yaw[None, :] if center_yaw.ndim == 1 else center_yaw
    else:
        raise ValueError("points' shape should be (N, 3) or (N, B, 3)")

    x = vectors[..., 0]
    y = vectors[..., 1]
    z = vectors[..., 2]

    # Calculate the horizontal angle (yaw/azimuth), assuming the Z-axis is forward.
    azimuth = torch.atan2(x, z) * (180 / math.pi)

    # Calculate the vertical angle (pitch/elevation).
    elevation = torch.atan2(y, torch.sqrt(x**2 + z**2)) * (180 / math.pi)

    # Calculate the angular difference from the center view direction (handling angle wrapping).
    diff_azimuth = azimuth - center_yaw
    diff_azimuth = torch.remainder(diff_azimuth + 180, 360) - 180

    diff_elevation = elevation - center_pitch
    diff_elevation = torch.remainder(diff_elevation + 180, 360) - 180

    # Check if within FOV
    in_fov_h = diff_azimuth.abs() < fov_half_h
    in_fov_v = diff_elevation.abs() < fov_half_v

    return in_fov_h & in_fov_v


def calculate_fov_overlap_similarity(
    w2c_matrix_curr: torch.Tensor,
    w2c_matrix_hist: torch.Tensor,
    fov_h_deg: float = 105.0,
    fov_v_deg: float = 75.0,
    device=None,
    points_local=None,
) -> float:
    """
    Calculate the Field-of-View (FOV) overlap similarity between two W2C poses using Monte Carlo sampling.

    Similarity = (Number of points in Curr_FOV ∩ Hist_FOV) / (Number of points in Curr_FOV).

    :param w2c_matrix_curr: The (4, 4) W2C matrix for the current frame.
    :param w2c_matrix_hist: The (4, 4) W2C matrix for the historical frame.
    :param num_samples, radius, fov_h_deg, fov_v_deg: Sampling and FOV parameters.
    :return: The overlap ratio (a float between 0.0 and 1.0).
    """
    w2c_matrix_curr = torch.tensor(w2c_matrix_curr, device=device)
    w2c_matrix_hist = torch.tensor(w2c_matrix_hist, device=device)

    c2w_matrix_curr = torch.linalg.inv(w2c_matrix_curr)
    c2w_matrix_hist = torch.linalg.inv(w2c_matrix_hist)
    C_inv = w2c_matrix_curr

    w2c_matrix_curr = torch.linalg.inv(C_inv @ c2w_matrix_curr)
    w2c_matrix_hist = torch.linalg.inv(C_inv @ c2w_matrix_hist)

    R_curr, t_curr = w2c_matrix_curr[:3, :3], w2c_matrix_curr[:3, 3]
    R_hist, t_hist = w2c_matrix_hist[:3, :3], w2c_matrix_hist[:3, 3]
    P_w_curr = -R_curr.T @ t_curr
    P_w_hist = -R_hist.T @ t_hist

    # pitch, yaw
    pitch_curr, yaw_curr = rotation_matrix_to_angles(R_curr)
    pitch_hist, yaw_hist = rotation_matrix_to_angles(R_hist)

    fov_half_h = torch.tensor(fov_h_deg / 2.0, device=device)
    fov_half_v = torch.tensor(fov_v_deg / 2.0, device=device)

    # move to P_w_curr (N, 3)
    points_world = points_local + P_w_curr[None, :]

    in_fov_curr = is_inside_fov_3d_hv(
        points_world,
        P_w_curr[None, :],
        pitch_curr[None],
        yaw_curr[None],
        fov_half_h,
        fov_half_v,
    )

    # compute based on angle
    in_fov_hist = is_inside_fov_3d_hv(
        points_world,
        P_w_hist[None, :],
        pitch_hist[None],
        yaw_hist[None],
        fov_half_h,
        fov_half_v,
    )

    # compute based on distance
    dist = torch.norm(points_world - P_w_hist.reshape(1, -1), dim=1) < 8.0
    in_fov_hist = in_fov_hist.bool() & dist.reshape(1, -1).bool()

    overlap_count = (in_fov_curr.bool() & in_fov_hist.bool()).sum().float()
    fov_curr_count = in_fov_curr.sum().float()

    if fov_curr_count == 0:
        return 0.0

    overlap_ratio = overlap_count / fov_curr_count

    return overlap_ratio.item()


def select_aligned_memory_frames(
    w2c_list: List[np.ndarray],
    current_frame_idx: int,
    memory_frames: int,
    temporal_context_size: int,
    pred_latent_size: int,
    pos_weight: float = 1.0,
    ang_weight: float = 1.0,
    device=None,
    points_local=None,
    debug_log_path=None,
    current_chunk_i=None,
    current_chunk_action=None,
) -> List[int]:
    """
    Selects memory and context frames for a given frame based on a four-frame segment distance calculation.

    :param w2c_list: List of all N 4x4 World-to-Camera (W2C) extrinsic matrices (np.ndarray).
    :param current_frame_idx: The index of the current frame to be processed.
    :param memory_frames: The total number of memory frames to select.
    :param context_size: The total number of context frames to select.
    :param pos_weight: The weight applied to the spatial (position) distance component.
    :param ang_weight: The weight applied to the angular distance component.

    :return: List[int]: A list containing the indices of the selected memory frames and context frames.
    """
    if current_frame_idx <= memory_frames:
        return list(range(0, current_frame_idx))

    num_total_frames = len(w2c_list)
    if current_frame_idx >= num_total_frames or current_frame_idx < 3:
        raise ValueError(
            f"The current frame index must be within the valid range of w2c_list and must be at least 3."
            f"{current_frame_idx}, {len(w2c_list)}"
        )

    start_context_idx = max(0, current_frame_idx - temporal_context_size)
    context_frames_indices = list(range(start_context_idx, current_frame_idx))

    candidate_distances = []
    query_clip_indices = list(
        range(
            current_frame_idx,
            (
                current_frame_idx + pred_latent_size
                if current_frame_idx + pred_latent_size <= num_total_frames
                else num_total_frames
            ),
        )
    )

    historical_clip_indices = list(
        range(4, current_frame_idx - temporal_context_size, 4)
    )
    ##一个可学习的检索器
    include_first_chunk = (
        os.getenv("ORIG_FOV_FIXED_FIRST_CHUNK", "0").strip().lower()
        in {"1", "true", "yes"}
    )
    memory_frames_indices = [0, 1, 2, 3] if include_first_chunk else []
    memory_frames = memory_frames - temporal_context_size

    for hist_idx in historical_clip_indices:
        total_dist = 0
        hist_w2c_1 = w2c_list[hist_idx]
        hist_w2c_2 = w2c_list[hist_idx + 2]
        for query_idx in query_clip_indices:
            dist_1_for_query_idx = 1.0 - calculate_fov_overlap_similarity(
                w2c_list[query_idx],
                hist_w2c_1,
                fov_h_deg=60.0,
                fov_v_deg=35.0,
                device=device,
                points_local=points_local,
            )
            dist_2_for_query_idx = 1.0 - calculate_fov_overlap_similarity(
                w2c_list[query_idx],
                hist_w2c_2,
                fov_h_deg=60.0,
                fov_v_deg=35.0,
                device=device,
                points_local=points_local,
            )
            dist_for_query_idx = (dist_1_for_query_idx + dist_2_for_query_idx) / 2.0
            total_dist += dist_for_query_idx

        final_clip_distance = total_dist / len(query_clip_indices)
        candidate_distances.append((hist_idx, final_clip_distance))

    candidate_distances.sort(key=lambda x: x[1])

    if debug_log_path is not None:
        os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)
        with open(debug_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"[Chunk Similarity] chunk_i={current_chunk_i} current_frame_idx={current_frame_idx} "
                f"query_clip_indices={query_clip_indices} current_chunk_action={current_chunk_action}\n"
            )
            for hist_idx, final_clip_distance in candidate_distances:
                similarity = 1.0 - final_clip_distance
                hist_chunk_i = hist_idx // 4
                f.write(
                    f"  hist_chunk_start={hist_idx} hist_chunk_i={hist_chunk_i} "
                    f"similarity={similarity:.6f} "
                    f"distance={final_clip_distance:.6f}\n"
                )

    for start_idx, _ in candidate_distances:
        # check the memory frame number
        if len(memory_frames_indices) >= memory_frames:
            break

        if start_idx not in memory_frames_indices:
            memory_frames_indices.extend(range(start_idx, start_idx + 4))

    # exclude the repeated frames
    selected_frames_set = set(context_frames_indices)
    selected_frames_set.update(memory_frames_indices)

    final_selected_frames = sorted(list(selected_frames_set))

    return final_selected_frames


def calculate_chunk_fov_similarity(
    w2c_list: List[np.ndarray],
    query_clip_indices: List[int],
    hist_idx: int,
    device=None,
    points_local=None,
    fov_h_deg: float = 60.0,
    fov_v_deg: float = 35.0,
) -> float:
    total_dist = 0.0
    hist_w2c_1 = w2c_list[hist_idx]
    hist_w2c_2 = w2c_list[hist_idx + 2]

    for query_idx in query_clip_indices:
        dist_1_for_query_idx = 1.0 - calculate_fov_overlap_similarity(
            w2c_list[query_idx],
            hist_w2c_1,
            fov_h_deg=fov_h_deg,
            fov_v_deg=fov_v_deg,
            device=device,
            points_local=points_local,
        )
        dist_2_for_query_idx = 1.0 - calculate_fov_overlap_similarity(
            w2c_list[query_idx],
            hist_w2c_2,
            fov_h_deg=fov_h_deg,
            fov_v_deg=fov_v_deg,
            device=device,
            points_local=points_local,
        )
        dist_for_query_idx = (dist_1_for_query_idx + dist_2_for_query_idx) / 2.0
        total_dist += dist_for_query_idx

    final_clip_distance = total_dist / len(query_clip_indices)
    return 1.0 - final_clip_distance
