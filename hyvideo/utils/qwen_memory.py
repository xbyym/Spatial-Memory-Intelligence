"""Immediate Qwen memory: quality, view classes, sparsity, and retrieval.

Invalid model decisions skip the corresponding operation. Infrastructure errors
remain fatal.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

import imageio.v2 as imageio
import torch

from .qwen_geometry import QwenGeometry


class QwenMemory(QwenGeometry):
    def __init__(self, points_local, device, output_dir, judge_url, min_vq=6):
        self.points_local = points_local
        self.execution_device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.judge_url = judge_url
        self.min_vq = min_vq
        self.chunk_latent_frames = 4
        self.chunk_geometry_bank = {}
        self.members = []
        self.clusters = []
        self.requests = []

    def request(self, task, payload, max_new_tokens):
        body = json.dumps(dict(task=task, payload=payload,
                               max_new_tokens=max_new_tokens), ensure_ascii=False).encode()
        request = Request(self.judge_url, data=body,
                          headers={'Content-Type': 'application/json; charset=utf-8'})
        with build_opener(ProxyHandler({})).open(request, timeout=180) as response:
            result = json.load(response)
        if not result['ok']:
            raise RuntimeError(result)
        self.requests.append(dict(task=task, payload=payload, response=result))
        return result['result']

    def payload(self, chunk_i, last=False):
        record = self.chunk_geometry_bank[chunk_i]
        payload = dict(chunk_i=chunk_i, cluster_id=record['cluster_id'],
                       frames=[record['last_frame' if last else 'mid_frame']])
        if record['visual_quality_score'] is not None:
            payload['visual_quality_score'] = record['visual_quality_score']
        if record['action'] is not None:
            payload['action'] = record['action']
        return payload

    def score_chunks(self, query_indices, viewmats, include=(), exclude=()):
        candidates = list(dict.fromkeys(self.members + list(include)))
        scores = []
        for chunk_i in candidates:
            record = self.chunk_geometry_bank[chunk_i]
            if chunk_i in exclude or record['removed']:
                continue
            overlap = self._compute_chunk_query_view_overlap(
                chunk_i, query_indices, viewmats, self.execution_device)
            scores.append(dict(chunk_i=chunk_i, cluster_id=record['cluster_id'],
                               overlap=overlap))
        return sorted(scores, key=lambda row: -row['overlap'])

    def refresh_prototype(self, cluster, force=False):
        if cluster['status'] == 'confirmed' and not force:
            return
        scores = []
        for chunk_i in cluster['members']:
            poses = self.chunk_geometry_bank[chunk_i]['worldplay_viewmats'].numpy()
            remaining = [i for i in cluster['members'] if i != chunk_i]
            overlap = self._compute_members_union_query_view_overlap(
                remaining, list(range(len(poses))), poses, self.execution_device)
            scores.append((overlap['coverage'] if overlap else 0.0, chunk_i))
        coverage, chunk_i = min(scores, key=lambda row: (-row[0], row[1]))
        cluster['prototype'] = chunk_i
        cluster['status'] = ('confirmed' if len(cluster['members']) >= 7
                             and coverage >= 0.95 else 'dynamic')

    def candidate_classes(self, chunk_i):
        record = self.chunk_geometry_bank[chunk_i]
        poses = record['worldplay_viewmats'].numpy()
        local = [i for i in [chunk_i - 2, chunk_i - 1]
                 if i >= 0 and not self.chunk_geometry_bank[i]['removed']]
        scores = self.score_chunks(list(range(len(poses))), poses, include=local)
        local_scores = [dict(row, local=True) for row in scores if row['chunk_i'] in local]
        extra_scores = [dict(row, local=False) for row in scores if row['chunk_i'] not in local][:6]
        votes = {}
        for rank, row in enumerate(local_scores + extra_scores):
            class_id = row['cluster_id']
            if class_id is None:
                continue
            vote = votes.setdefault(class_id, dict(overlap=0., rank=rank, local=False))
            vote['overlap'] = max(vote['overlap'], row['overlap'])
            vote['local'] |= row['local']
        if votes:
            return sorted(votes, key=lambda i: (-votes[i]['overlap'],
                          not votes[i]['local'], votes[i]['rank'], i))[:3]
        recent = []
        for i in range(chunk_i - 1, -1, -1):
            old = self.chunk_geometry_bank[i]
            class_id = old['cluster_id']
            if not old['removed'] and class_id is not None and class_id not in recent:
                recent.append(class_id)
            if len(recent) == 2:
                break
        return recent

    def sparse(self, cluster):
        members = cluster['members'][:]
        if len(members) < 7:
            return
        candidates = [members[-2], members[-3]]
        scenarios = []
        for index, candidate in enumerate(candidates):
            for removed in ([[]] if index == 0 else [[], [candidates[0]]]):
                remaining = [i for i in members if i != candidate and i not in removed]
                poses = self.chunk_geometry_bank[candidate]['worldplay_viewmats'].numpy()
                coverage = self._compute_members_union_query_view_overlap(
                    remaining, list(range(len(poses))), poses, self.execution_device)
                payload = dict(cluster_id=cluster['id'], candidate_chunk_i=candidate,
                    candidate_chunk=self.payload(candidate), remaining_chunk_ids=remaining,
                    remaining_chunks=[self.payload(i) for i in remaining],
                    coverage_by_remaining_chunks=coverage['coverage'],
                    scenario_assumed_removed_chunk_ids=removed,
                    member_position=len(members) - index - 1, member_from_end=-index - 2)
                scenarios.append((candidate, tuple(removed), payload))

        def judge(scenario):
            candidate, removed, payload = scenario
            return (candidate, removed), self.request('sparse', payload, 128)

        with ThreadPoolExecutor(max_workers=len(scenarios)) as executor:
            responses = dict(executor.map(judge, scenarios))
        removed = []
        for candidate in candidates:
            result = responses[candidate, tuple(sorted(removed))]
            if not result.get('skipped') and result.get('action') == 'remove':
                removed.append(candidate)
        for candidate in removed:
            self.chunk_geometry_bank[candidate]['removed'] = True
            self.chunk_geometry_bank[candidate]['cluster_id'] = None
        cluster['members'] = [i for i in members if i not in removed]
        self.members = [i for i in self.members if i not in removed]
        if cluster['prototype'] in removed:
            self.refresh_prototype(cluster, force=True)
        elif cluster['status'] != 'confirmed':
            self.refresh_prototype(cluster)

    def add(self, chunk_i, video, viewmats, action):
        """video is the reference VAE's float32 C,F,H,W output in [0,1]."""
        rgb = (video.detach().cpu().permute(1, 2, 3, 0) * 255).clamp(0, 255).to(torch.uint8)
        paths = {}
        for name, index in [('mid', len(rgb) // 2), ('last', len(rgb) - 1)]:
            path = self.output_dir / f'chunk_{chunk_i:03d}_{name}.png'
            imageio.imwrite(path, rgb[index].numpy())
            paths[name + '_frame'] = str(path.resolve())
        record = dict(paths, worldplay_viewmats=viewmats.detach().cpu().clone(),
                      action=None if action is None else torch.as_tensor(action).tolist(),
                      cluster_id=None, visual_quality_score=None, removed=False)
        self.chunk_geometry_bank[chunk_i] = record
        self.update(chunk_i)

    def update(self, chunk_i):
        record = self.chunk_geometry_bank[chunk_i]
        quality = self.request('visual_quality', dict(generated_chunk=self.payload(chunk_i)), 64)
        if quality.get('skipped'):
            self.save_trace(chunk_i, 'update', dict(skipped=True, reason=quality['reason']))
            return
        record['visual_quality_score'] = max(1, min(10, int(round(float(quality['visual_quality_score'])))))
        if record['visual_quality_score'] < self.min_vq:
            record['removed'] = True
            self.save_trace(chunk_i, 'update')
            return
        class_ids = self.candidate_classes(chunk_i)
        payloads, prototypes = [], []
        for class_id in class_ids:
            cluster = dict(self.clusters[class_id])
            self.refresh_prototype(cluster)
            prototypes.append(cluster)
            prototype = self.payload(cluster['prototype'])
            overlap = self._compute_chunk_view_overlap(chunk_i, cluster['prototype'], self.execution_device)
            prototype.update(cluster_id=class_id, fov_overlap=overlap)
            payloads.append(dict(current_chunk=self.payload(chunk_i), prototype=prototype,
                                 class_id=class_id, cluster_id=class_id, prototype_fov_overlap=overlap))
        results = []
        if payloads:
            with ThreadPoolExecutor(max_workers=len(payloads)) as executor:
                results = list(executor.map(lambda p: self.request('create', p, 128), payloads))
        if any(result.get('skipped') for result in results):
            self.save_trace(chunk_i, 'update', dict(skipped=True, reason='invalid_model_output'))
            return
        for class_id, cluster in zip(class_ids, prototypes):
            self.clusters[class_id].update(cluster)
        matched = next((i for i, result in zip(class_ids, results) if result['prototype_match']), None)
        if matched is None:
            matched = len(self.clusters)
            self.clusters.append(dict(id=matched, members=[chunk_i], prototype=chunk_i, status='tentative'))
            record['cluster_id'] = matched
        else:
            cluster = self.clusters[matched]
            cluster['members'].append(chunk_i)
            record['cluster_id'] = matched
            self.refresh_prototype(cluster)
            self.sparse(cluster)
        self.members.append(chunk_i)
        self.save_trace(chunk_i, 'update')

    def select(self, chunk_i, viewmats, action):
        local = list(range(max(0, chunk_i - 2), chunk_i))
        query_indices = list(range(chunk_i * 4, (chunk_i + 1) * 4))
        poses = viewmats.detach().cpu().numpy()
        scores = self.score_chunks(query_indices, poses, exclude=local)
        scores.sort(key=lambda row: (-row['overlap'], row['chunk_i']))
        candidates = []
        for row in scores:
            payload = self.payload(row['chunk_i'], last=True)
            if payload['visual_quality_score'] < self.min_vq:
                continue
            payload.update(class_id=payload['cluster_id'], fov_overlap=row['overlap'],
                           selection_role_hint='coverage_candidate', frame_role='coverage_candidate_last_frame',
                           option_label=chr(65 + len(candidates)), select_source=None)
            candidates.append(payload)
            if len(candidates) == 7:
                break
        selected = []
        if candidates:
            current = self.payload(local[-1], last=True)
            current['frame_role'] = 'current_chunk_last_frame'
            motion = self._summarize_qwen_motion(action, query_indices, poses)
            motion_text = self._format_qwen_target_motion_text(motion)
            response = self.request('select_coverage', dict(current_chunk_i=chunk_i,
                target_next_chunk_id=chunk_i, current_chunk=current,
                planned_camera_motion=motion_text, target_motion_text=motion_text,
                motion_summary=motion, coverage_candidates=candidates, candidates=candidates), 768)
            by_option = {p['option_label']: p['chunk_i'] for p in candidates}
            selected = list(dict.fromkeys(by_option[label] for label in response.get('selected_coverage', [])
                                          if label in by_option))[:2]
        chunks = sorted(set(local + selected))
        indices = [frame for i in chunks for frame in range(i * 4, (i + 1) * 4)]
        self.save_trace(chunk_i, 'select', dict(local_chunk_ids=local,
                        memory_chunk_ids=sorted(selected), selected_frame_indices=indices))
        return indices

    def save_trace(self, chunk_i, operation, extra=None):
        trace = dict(chunk_i=chunk_i, operation=operation, members=self.members,
                     clusters=self.clusters, requests=self.requests, **(extra or {}))
        path = self.output_dir / f'chunk_{chunk_i:03d}_{operation}.json'
        path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + '\n')
        self.requests = []
