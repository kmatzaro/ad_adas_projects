from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from perception.object_detection import DetectedObject


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    class_id: int
    missed: int = 0


class ObjectTracker:
    """
    Frame-to-frame IoU-based multi-object tracker.

    Each detected object is assigned a persistent integer track_id that
    survives across frames as long as a matching detection continues to appear.

    Algorithm:
    - Build an IoU matrix between existing tracks and new detections.
      Pairs with mismatched class_id are forced to 0.
    - Greedily assign the highest-IoU pair first (repeat until all pairs
      are below iou_threshold).
    - Unmatched detections start new tracks.
    - Unmatched tracks have their 'missed' counter incremented; tracks
      that exceed max_missed_frames are pruned.
    """

    def __init__(self, config: Dict):
        tracker_cfg = config.get('object_tracker', {})
        self.iou_threshold: float = tracker_cfg.get('iou_threshold', 0.3)
        self.max_missed: int = tracker_cfg.get('max_missed_frames', 5)
        self._tracks: Dict[int, Track] = {}
        self._next_id: int = 1

    def update(self, detections: List[DetectedObject]) -> List[DetectedObject]:
        """
        Match detections to existing tracks, populate track_id on each
        DetectedObject, and return the same list.
        """
        if not self._tracks:
            for det in detections:
                det.track_id = self._next_id
                self._tracks[self._next_id] = Track(
                    track_id=self._next_id,
                    bbox=det.bbox,
                    class_id=det.class_id,
                )
                self._next_id += 1
            return detections

        track_ids = list(self._tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(detections)

        # Build IoU matrix (tracks × detections)
        iou_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)
        for r, tid in enumerate(track_ids):
            for c, det in enumerate(detections):
                if self._tracks[tid].class_id == det.class_id:
                    iou_matrix[r, c] = self._compute_iou(
                        self._tracks[tid].bbox, det.bbox
                    )

        # Greedy matching: pick highest-IoU pair repeatedly
        matched_tracks: set = set()
        matched_dets: set = set()

        if n_tracks > 0 and n_dets > 0:
            flat_order = np.argsort(iou_matrix, axis=None)[::-1]
            for flat_idx in flat_order:
                r, c = divmod(int(flat_idx), n_dets)
                if iou_matrix[r, c] < self.iou_threshold:
                    break
                if r in matched_tracks or c in matched_dets:
                    continue
                tid = track_ids[r]
                self._tracks[tid].bbox = detections[c].bbox
                self._tracks[tid].missed = 0
                detections[c].track_id = tid
                matched_tracks.add(r)
                matched_dets.add(c)

        # Unmatched detections → new tracks
        for c, det in enumerate(detections):
            if c not in matched_dets:
                det.track_id = self._next_id
                self._tracks[self._next_id] = Track(
                    track_id=self._next_id,
                    bbox=det.bbox,
                    class_id=det.class_id,
                )
                self._next_id += 1

        # Unmatched tracks → increment missed, prune if over threshold
        stale = []
        for r, tid in enumerate(track_ids):
            if r not in matched_tracks:
                self._tracks[tid].missed += 1
                if self._tracks[tid].missed > self.max_missed:
                    stale.append(tid)
        for tid in stale:
            del self._tracks[tid]

        return detections

    @staticmethod
    def _compute_iou(boxA: Tuple, boxB: Tuple) -> float:
        """Intersection-over-Union for two (x1, y1, x2, y2) boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return 0.0
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter / float(areaA + areaB - inter)