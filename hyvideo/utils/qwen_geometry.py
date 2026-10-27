"""FoV and motion arithmetic matching the frozen HY1.5 Qwen reference.

Derived from the 2026-09-26 runtime under the upstream Tencent Hunyuan license.
Preserves operation order and sampling. Invalid inputs fail immediately.
"""
from typing import List

import numpy as np
import torch

from hyvideo.utils.retrieval_context import (
    calculate_fov_overlap_similarity,
    is_inside_fov_3d_hv,
    rotation_matrix_to_angles,
)

def calculate_chunk_fov_similarity(w2c_list: List[np.ndarray], query_clip_indices: List[int], hist_idx: int, device=None, points_local=None, fov_h_deg: float=60.0, fov_v_deg: float=35.0) -> float:
    total_dist = 0.0
    hist_w2c_1 = w2c_list[hist_idx]
    hist_w2c_2 = w2c_list[hist_idx + 2]
    for query_idx in query_clip_indices:
        dist_1_for_query_idx = 1.0 - calculate_fov_overlap_similarity(w2c_list[query_idx], hist_w2c_1, fov_h_deg=fov_h_deg, fov_v_deg=fov_v_deg, device=device, points_local=points_local)
        dist_2_for_query_idx = 1.0 - calculate_fov_overlap_similarity(w2c_list[query_idx], hist_w2c_2, fov_h_deg=fov_h_deg, fov_v_deg=fov_v_deg, device=device, points_local=points_local)
        dist_for_query_idx = (dist_1_for_query_idx + dist_2_for_query_idx) / 2.0
        total_dist += dist_for_query_idx
    final_clip_distance = total_dist / len(query_clip_indices)
    return 1.0 - final_clip_distance

class QwenGeometry:

    def _compute_chunk_query_view_overlap(self, chunk_i, query_clip_indices, viewmats_np, device):
        chunk_record = self.chunk_geometry_bank.get(int(chunk_i))
        if chunk_record is None:
            return None
        chunk_viewmats = chunk_record.get('worldplay_viewmats')
        if chunk_viewmats is None or chunk_viewmats.shape[0] == 0:
            return None
        merged_viewmats = torch.cat([chunk_viewmats.detach().cpu(), torch.tensor(viewmats_np[query_clip_indices]).cpu()], dim=0)
        merged_np = merged_viewmats.numpy()
        query_indices = list(range(chunk_viewmats.shape[0], chunk_viewmats.shape[0] + len(query_clip_indices)))
        return float(calculate_chunk_fov_similarity(merged_np, query_indices, 0, device=device, points_local=self.points_local))

    def _compute_members_union_query_view_overlap(self, member_chunk_ids, query_clip_indices, viewmats_np, device, fov_h_deg: float=60.0, fov_v_deg: float=35.0):
        if self.points_local is None:
            return None
        member_chunk_ids = [int(x) for x in member_chunk_ids or []]
        if not member_chunk_ids or not query_clip_indices:
            return None
        hist_frames = []
        covering_member_chunk_ids = []
        for member_chunk_i in member_chunk_ids:
            member_record = self.chunk_geometry_bank.get(int(member_chunk_i))
            if member_record is None:
                continue
            member_viewmats = member_record.get('worldplay_viewmats')
            if member_viewmats is None or member_viewmats.shape[0] == 0:
                continue
            representative_indices = [0]
            if member_viewmats.shape[0] > 2:
                representative_indices.append(2)
            elif member_viewmats.shape[0] > 1:
                representative_indices.append(member_viewmats.shape[0] - 1)
            representative_indices = list(dict.fromkeys(representative_indices))
            member_viewmats_cpu = member_viewmats.detach().cpu()
            for frame_idx in representative_indices:
                hist_frames.append(member_viewmats_cpu[int(frame_idx)].numpy())
            covering_member_chunk_ids.append(int(member_chunk_i))
        if not hist_frames:
            return None
        points_local = self.points_local.to(device=device)
        fov_half_h = torch.tensor(float(fov_h_deg) / 2.0, device=device)
        fov_half_v = torch.tensor(float(fov_v_deg) / 2.0, device=device)
        hist_w2c_mats = [torch.as_tensor(frame, device=device, dtype=torch.float32) for frame in hist_frames]
        coverages = []
        with torch.no_grad():
            for query_idx in query_clip_indices:
                query_w2c = torch.as_tensor(viewmats_np[int(query_idx)], device=device, dtype=torch.float32)
                query_c2w = torch.linalg.inv(query_w2c)
                query_relative_w2c = torch.linalg.inv(query_w2c @ query_c2w)
                R_query = query_relative_w2c[:3, :3]
                t_query = query_relative_w2c[:3, 3]
                P_w_query = -R_query.T @ t_query
                (pitch_query, yaw_query) = rotation_matrix_to_angles(R_query)
                points_world = points_local + P_w_query[None, :]
                in_fov_query = is_inside_fov_3d_hv(points_world, P_w_query[None, :], pitch_query[None], yaw_query[None], fov_half_h, fov_half_v).bool()
                fov_query_count = in_fov_query.sum().float()
                if float(fov_query_count.item()) <= 0.0:
                    coverages.append(0.0)
                    continue
                union_hist = torch.zeros_like(in_fov_query, dtype=torch.bool)
                for hist_w2c in hist_w2c_mats:
                    hist_c2w = torch.linalg.inv(hist_w2c)
                    hist_relative_w2c = torch.linalg.inv(query_w2c @ hist_c2w)
                    R_hist = hist_relative_w2c[:3, :3]
                    t_hist = hist_relative_w2c[:3, 3]
                    P_w_hist = -R_hist.T @ t_hist
                    (pitch_hist, yaw_hist) = rotation_matrix_to_angles(R_hist)
                    in_fov_hist = is_inside_fov_3d_hv(points_world, P_w_hist[None, :], pitch_hist[None], yaw_hist[None], fov_half_h, fov_half_v).bool()
                    dist = torch.norm(points_world - P_w_hist.reshape(1, -1), dim=1) < 8.0
                    union_hist = union_hist | in_fov_hist & dist.reshape(1, -1).bool()
                overlap_count = (in_fov_query & union_hist).sum().float()
                coverages.append(float((overlap_count / fov_query_count).item()))
        if not coverages:
            return None
        return {'coverage': float(sum(coverages) / len(coverages)), 'query_frame_coverages': coverages, 'covering_member_chunk_ids': covering_member_chunk_ids, 'covering_frame_count': int(len(hist_frames))}

    def _decode_worldplay_action_label(self, action_id):
        action_id = int(action_id)
        trans_id = action_id // 9
        rot_id = action_id % 9
        trans_names = {0: 'none', 1: 'forward', 2: 'backward', 3: 'right', 4: 'left', 5: 'forward+right', 6: 'forward+left', 7: 'backward+right', 8: 'backward+left'}
        rot_names = {0: 'none', 1: 'yaw_right', 2: 'yaw_left', 3: 'pitch_up', 4: 'pitch_down', 5: 'yaw_right+pitch_up', 6: 'yaw_right+pitch_down', 7: 'yaw_left+pitch_up', 8: 'yaw_left+pitch_down'}
        return {'raw': action_id, 'translation': trans_names.get(trans_id, f'unknown_{trans_id}'), 'rotation': rot_names.get(rot_id, f'unknown_{rot_id}')}

    def _summarize_qwen_motion(self, current_chunk_action=None, query_clip_indices=None, viewmats_np=None):
        raw_actions = list(current_chunk_action or [])
        decoded_steps = [self._decode_worldplay_action_label(x) for x in raw_actions]
        counts = {}
        forward_units = 0.0
        right_units = 0.0
        yaw_degrees = 0.0
        pitch_degrees = 0.0
        roll_degrees = 0.0
        motion_source = 'action_ids' if raw_actions else 'none'
        for step in decoded_steps:
            trans = step.get('translation', 'unknown')
            rot = step.get('rotation', 'unknown')
            key = f'{trans}|{rot}'
            counts[key] = counts.get(key, 0) + 1
            trans_parts = set(str(trans).split('+'))
            rot_parts = set(str(rot).split('+'))
            if 'forward' in trans_parts:
                forward_units += 0.08
            if 'backward' in trans_parts:
                forward_units -= 0.08
            if 'right' in trans_parts:
                right_units += 0.08
            if 'left' in trans_parts:
                right_units -= 0.08
            if 'yaw_right' in rot_parts:
                yaw_degrees += 3.0
            if 'yaw_left' in rot_parts:
                yaw_degrees -= 3.0
            if 'pitch_up' in rot_parts:
                pitch_degrees += 3.0
            if 'pitch_down' in rot_parts:
                pitch_degrees -= 3.0
        screen_effects = []
        dominant = None
        if counts:
            dominant = max(counts.items(), key=lambda x: x[1])[0]
        pose_delta = None
        pose_estimated_control_delta = None
        if viewmats_np is not None and query_clip_indices and (len(query_clip_indices) > 0):
            start_idx = max(0, int(query_clip_indices[0]) - 1)
            end_idx = int(query_clip_indices[-1])
            start_c2w = np.linalg.inv(np.asarray(viewmats_np[start_idx]))
            end_c2w = np.linalg.inv(np.asarray(viewmats_np[end_idx]))
            delta_t = end_c2w[:3, 3] - start_c2w[:3, 3]
            delta_local = start_c2w[:3, :3].T @ delta_t
            rel_rot = start_c2w[:3, :3].T @ end_c2w[:3, :3]
            cos_theta = (float(np.trace(rel_rot)) - 1.0) * 0.5
            cos_theta = max(-1.0, min(1.0, cos_theta))
            rot_angle_deg = float(np.degrees(np.arccos(cos_theta)))
            start_R = torch.as_tensor(np.asarray(viewmats_np[start_idx])[:3, :3], device=self.execution_device, dtype=torch.float32)
            end_R = torch.as_tensor(np.asarray(viewmats_np[end_idx])[:3, :3], device=self.execution_device, dtype=torch.float32)
            (start_pitch, start_yaw) = rotation_matrix_to_angles(start_R)
            (end_pitch, end_yaw) = rotation_matrix_to_angles(end_R)

            def _wrap_delta_degrees(value):
                return (float(value) + 180.0) % 360.0 - 180.0
            yaw_delta = _wrap_delta_degrees(float(end_yaw.detach().cpu().item()) - float(start_yaw.detach().cpu().item()))
            pitch_delta = _wrap_delta_degrees(float(end_pitch.detach().cpu().item()) - float(start_pitch.detach().cpu().item()))
            pose_estimated_control_delta = {'forward_units': round(float(delta_local[2]), 4), 'right_units': round(float(delta_local[0]), 4), 'up_units': round(float(delta_local[1]), 4), 'yaw_degrees': round(float(yaw_delta), 4), 'pitch_degrees': round(float(pitch_delta), 4), 'roll_degrees': 0.0}
            pose_delta = {'from_pose_index': int(start_idx), 'to_pose_index': int(end_idx), 'translation_norm': float(np.linalg.norm(delta_t)), 'translation_world': [float(x) for x in delta_t.tolist()], 'translation_local_right_up_forward': [float(x) for x in delta_local.tolist()], 'yaw_delta_degrees': float(yaw_delta), 'pitch_delta_degrees': float(pitch_delta), 'rotation_angle_degrees': rot_angle_deg}
        if not raw_actions and pose_estimated_control_delta is not None:
            forward_units = float(pose_estimated_control_delta.get('forward_units', 0.0))
            right_units = float(pose_estimated_control_delta.get('right_units', 0.0))
            yaw_degrees = float(pose_estimated_control_delta.get('yaw_degrees', 0.0))
            pitch_degrees = float(pose_estimated_control_delta.get('pitch_degrees', 0.0))
            roll_degrees = float(pose_estimated_control_delta.get('roll_degrees', 0.0))
            motion_source = 'pose_delta'
        if yaw_degrees < -1e-06:
            screen_effects.append('camera_turn_left: previously visible objects drift right in the image; new content enters from the left')
        elif yaw_degrees > 1e-06:
            screen_effects.append('camera_turn_right: previously visible objects drift left in the image; new content enters from the right')
        if forward_units > 1e-06:
            screen_effects.append('camera_moves_forward: nearby objects expand and move outward')
        elif forward_units < -1e-06:
            screen_effects.append('camera_moves_backward: nearby objects shrink and move inward')
        if right_units > 1e-06:
            screen_effects.append('camera_strafes_right: scene content drifts left in the image')
        elif right_units < -1e-06:
            screen_effects.append('camera_strafes_left: scene content drifts right in the image')
        return {'raw_action_ids': raw_actions, 'decoded_steps': decoded_steps, 'counts': counts, 'dominant_step': dominant, 'motion_source': motion_source, 'estimated_control_delta': {'forward_units': round(float(forward_units), 4), 'right_units': round(float(right_units), 4), 'up_units': round(float((pose_estimated_control_delta or {}).get('up_units', 0.0)), 4), 'yaw_degrees': round(float(yaw_degrees), 4), 'pitch_degrees': round(float(pitch_degrees), 4), 'roll_degrees': round(float(roll_degrees), 4)}, 'screen_effect': screen_effects, 'pose_delta': pose_delta, 'note': 'Action ids decode as translation_id * 9 + rotation_id. If action ids are unavailable, estimated_control_delta is derived from camera pose delta. Positive yaw means camera turns right; negative yaw means camera turns left. For yaw, the screen drift of existing objects is opposite to the camera turn direction.'}

    def _format_qwen_target_motion_text(self, motion_summary):
        estimated = (motion_summary or {}).get('estimated_control_delta', {}) or {}
        yaw_degrees = float(estimated.get('yaw_degrees', 0.0))
        forward_units = float(estimated.get('forward_units', 0.0))
        right_units = float(estimated.get('right_units', 0.0))
        up_units = float(estimated.get('up_units', 0.0))
        pitch_degrees = float(estimated.get('pitch_degrees', 0.0))
        roll_degrees = float(estimated.get('roll_degrees', 0.0))
        motion_tokens = []
        if abs(yaw_degrees) >= 5.0:
            motion_tokens.append('turn_right' if yaw_degrees > 0 else 'turn_left')
        if abs(pitch_degrees) >= 5.0:
            motion_tokens.append('tilt_up' if pitch_degrees > 0 else 'tilt_down')
        translation_axes = {'move_right': float(right_units), 'move_up': float(up_units), 'move_forward': float(forward_units)}
        (dominant_axis, dominant_value) = max(translation_axes.items(), key=lambda item: abs(float(item[1])))
        if abs(float(dominant_value)) > 1e-06:
            if dominant_axis == 'move_right':
                motion_tokens.append('move_right' if dominant_value > 0 else 'move_left')
            elif dominant_axis == 'move_up':
                motion_tokens.append('move_up' if dominant_value > 0 else 'move_down')
            elif dominant_axis == 'move_forward':
                motion_tokens.append('move_forward' if dominant_value > 0 else 'move_backward')
        if not motion_tokens:
            motion_tokens.append('mostly_static')
        camera_motion_text = f"camera_motion={'+'.join(motion_tokens)}"
        return f'{camera_motion_text}; local_translation(right={right_units:.4f}, up={up_units:.4f}, forward={forward_units:.4f}); local_rotation(yaw={yaw_degrees:.4f}deg, pitch={pitch_degrees:.4f}deg, roll={roll_degrees:.4f}deg)'

    def _compute_chunk_view_overlap(self, query_chunk_i, candidate_chunk_i, device):
        query_record = self.chunk_geometry_bank.get(query_chunk_i)
        candidate_record = self.chunk_geometry_bank.get(candidate_chunk_i)
        if query_record is None or candidate_record is None:
            return 0.0
        query_viewmats = query_record.get('worldplay_viewmats')
        candidate_viewmats = candidate_record.get('worldplay_viewmats')
        if query_viewmats is None or candidate_viewmats is None:
            return 0.0
        merged_viewmats = torch.cat([candidate_viewmats, query_viewmats], dim=0)
        merged_np = merged_viewmats.cpu().detach().numpy()
        query_clip_indices = list(range(self.chunk_latent_frames, self.chunk_latent_frames * 2))
        return float(calculate_chunk_fov_similarity(merged_np, query_clip_indices, 0, device=device, points_local=self.points_local))

    def _compute_fov_union_coverage(self, query_chunk_i, member_chunk_ids, device=None):
        query_record = self.chunk_geometry_bank.get(int(query_chunk_i))
        if query_record is None:
            return 0.0
        query_viewmats = query_record.get('worldplay_viewmats')
        if query_viewmats is None or query_viewmats.shape[0] == 0:
            return 0.0
        points_local = getattr(self, 'points_local', None)
        if points_local is None:
            return 0.0
        if device is None:
            device = self.execution_device
        hist_viewmats = []
        for member_chunk_i in member_chunk_ids or []:
            member_record = self.chunk_geometry_bank.get(int(member_chunk_i))
            if member_record is None:
                continue
            member_viewmats = member_record.get('worldplay_viewmats')
            if member_viewmats is None or member_viewmats.shape[0] == 0:
                continue
            rep_indices = sorted({0, min(2, int(member_viewmats.shape[0]) - 1), int(member_viewmats.shape[0]) - 1})
            for rep_idx in rep_indices:
                hist_viewmats.append(member_viewmats[int(rep_idx)])
        if not hist_viewmats:
            return 0.0
        points_local = points_local.to(device=device, dtype=torch.float32)
        fov_half_h = torch.tensor(60.0 / 2.0, device=device, dtype=torch.float32)
        fov_half_v = torch.tensor(35.0 / 2.0, device=device, dtype=torch.float32)

        def _relative_w2c(anchor_w2c, target_w2c):
            anchor_w2c = torch.as_tensor(anchor_w2c, device=device, dtype=torch.float32)
            target_w2c = torch.as_tensor(target_w2c, device=device, dtype=torch.float32)
            target_c2w = torch.linalg.inv(target_w2c)
            return torch.linalg.inv(anchor_w2c @ target_c2w)

        def _camera_fov_mask(points_world, rel_w2c):
            R = rel_w2c[:3, :3]
            t = rel_w2c[:3, 3]
            center = -R.T @ t
            (pitch, yaw) = rotation_matrix_to_angles(R)
            mask = is_inside_fov_3d_hv(points_world, center[None, :], pitch[None], yaw[None], fov_half_h, fov_half_v).bool().reshape(-1)
            return (mask, center)
        coverages = []
        with torch.no_grad():
            for query_w2c in query_viewmats:
                query_w2c = torch.as_tensor(query_w2c, device=device, dtype=torch.float32)
                query_rel_w2c = _relative_w2c(query_w2c, query_w2c)
                query_R = query_rel_w2c[:3, :3]
                query_t = query_rel_w2c[:3, 3]
                query_center = -query_R.T @ query_t
                points_world = points_local + query_center[None, :]
                (query_mask, _) = _camera_fov_mask(points_world, query_rel_w2c)
                denom = query_mask.sum().float()
                if denom.item() <= 0:
                    continue
                union_mask = torch.zeros_like(query_mask, dtype=torch.bool)
                for hist_w2c in hist_viewmats:
                    hist_rel_w2c = _relative_w2c(query_w2c, hist_w2c)
                    (hist_mask, hist_center) = _camera_fov_mask(points_world, hist_rel_w2c)
                    hist_dist_mask = torch.norm(points_world - hist_center.reshape(1, -1), dim=1) < 8.0
                    union_mask |= hist_mask & hist_dist_mask.reshape(-1)
                coverage = (query_mask & union_mask).sum().float() / denom
                coverages.append(float(coverage.item()))
        if not coverages:
            return 0.0
        return float(sum(coverages) / len(coverages))
