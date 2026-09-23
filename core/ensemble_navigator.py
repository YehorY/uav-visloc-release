import numpy as np
import logging
from core.pathfinder import (
    find_photo_path_beam_search,
    pre_filter_nodes_global,
    find_map_paths_A_star_beam_search
)
from core.comparator import _get_path_properties, compare_graphs
from topsis.manual_topsis import apply_topsis

NEURAL_PATH_LENGTH = 16   # the Siamese net was trained on 16-point paths (32 inputs); see tools/generate_dataset.py


def resample_path(points: np.ndarray, n: int = NEURAL_PATH_LENGTH) -> np.ndarray:
    """Resample a polyline to exactly n points by arc length. Production routes have <= 10 vertices, while
    the net takes 16, so every route is resampled before scoring. Deterministic; degenerate paths repeat p0."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros((n, 2))
    if len(pts) == 1:
        return np.repeat(pts, n, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 1e-9:
        return np.repeat(pts[:1], n, axis=0)
    t = np.linspace(0.0, cum[-1], n)
    return np.stack([np.interp(t, cum, pts[:, 0]), np.interp(t, cum, pts[:, 1])], axis=1)


class EnsembleNavigator:
    def __init__(self, global_anchors_array, global_classes_array, neural_cfg=None):
        self.global_map = np.array(global_anchors_array)
        self.global_classes = np.array(global_classes_array)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("EnsembleNavigator")
        # Siamese similarity as a 4th TOPSIS criterion. Off unless `neural_comparator.enabled` is set, so the
        # geometric-only control stays reproducible. CPU on purpose: eval-mode MLP, deterministic, ~0.2 ms/candidate.
        self.neural = None
        self.neural_weight = 0.0
        cfg = neural_cfg or {}
        if cfg.get("enabled"):
            from core.neural_comparator import NeuralComparator
            self.neural = NeuralComparator(cfg["model_path"], device_type=cfg.get("device", "cpu"))
            self.neural_weight = float(cfg.get("weight", 0.2))
            self.logger.info(f"Neural criterion ON (weight {self.neural_weight}, model {cfg['model_path']})")

    def _neural_distances(self, photo_route_points, all_map_routes):
        """Embedding distance (cost criterion) per candidate, in candidate order."""
        cam = resample_path(photo_route_points)
        maps = [resample_path(self.global_map[route]) for route in all_map_routes]
        scored = self.neural.compare_batch(cam, maps)          # returns (idx, dist) sorted by distance
        out = np.zeros(len(maps), dtype=float)
        for idx, dist in scored:
            out[idx] = dist
        return out

    def evaluate_candidates(self, ref_photo_nodes_array, ref_photo_route, all_map_routes, expected_constellation=None, node_to_constellation=None, uncertainty_ratio=0.0, criteria=None):
        """LEVEL 2.3: `criteria` (ablation A2) = None for production multi-criteria TOPSIS, or a single
        criterion ["shape"] (angle_error) or ["ratio"] (edge-length ratio_error)."""
        def normalize_path(points: np.ndarray) -> np.ndarray:
            if len(points) == 0:
                return points
            centroid = np.mean(points, axis=0)
            centered = points - centroid
            max_val = np.max(np.abs(centered))
            if max_val > 0:
                return centered / max_val
            return centered

        photo_route_points = ref_photo_nodes_array[ref_photo_route]
        photo_norm_points = normalize_path(photo_route_points)
        photo_types = [False for _ in ref_photo_route]

        photo_features = _get_path_properties(photo_norm_points, photo_types)

        map_normalized_points_list = []
        map_types_list = []
        for route in all_map_routes:
            pts = self.global_map[route]
            map_normalized_points_list.append(normalize_path(pts))
            map_types_list.append([False for _ in route])

        candidates_scores = compare_graphs(
            photo_route=ref_photo_route,
            map_routes=all_map_routes,
            photo_features=photo_features,
            map_normalized_points_list=map_normalized_points_list,
            map_types_list=map_types_list
        )
        
        if not candidates_scores:
            return None

        if criteria:
            # LEVEL 2.3 / A2: single-criterion selection. TOPSIS with one cost criterion reduces exactly
            # to argmin with score (max - v) / (max - min); computed directly, which also avoids TOPSIS's
            # 0/0 when every candidate ties (then all score 1.0 and the first candidate wins).
            _cols = {"shape": "angle_error", "ratio": "ratio_error"}
            if len(criteria) != 1 or criteria[0] not in _cols:
                raise ValueError(f"criteria must be ['shape'] or ['ratio']; got {criteria!r}")
            _v = np.asarray([cand[_cols[criteria[0]]] for cand in candidates_scores], dtype=float)
            best_idx = int(np.argmin(_v))
            _span = float(_v.max() - _v.min())
            scores = (_v.max() - _v) / _span if _span > 0 else np.ones_like(_v)
            return self._package(candidates_scores, scores, best_idx)

        # Siamese embedding distance per candidate (cost). Computed over the candidates that survived
        # compare_graphs, in their order, so the column cannot be misaligned with the rows.
        neural_col = None
        if self.neural is not None:
            neural_col = self._neural_distances(photo_route_points, [cand['route'] for cand in candidates_scores])

        decision_matrix = []
        for _i, cand in enumerate(candidates_scores):
            shape_err = cand['angle_error']
            dist_err = cand['ratio_error']
            row = [shape_err, dist_err]

            # --- DYNAMIC CONSTELLATION AFFINITY ---
            if expected_constellation is not None and node_to_constellation is not None:
                # Count how many map nodes in this route belong to the expected constellation
                route = cand['route']
                hit_count = sum(1 for node_id in route if node_to_constellation.get(node_id) == expected_constellation)
                
                # Calculate affinity percentage (0.0 to 1.0). Higher is better.
                affinity = hit_count / len(route) if len(route) > 0 else 0.0
                row.append(affinity)
            # --------------------------------------

            if neural_col is not None:
                row.append(float(neural_col[_i]))

            decision_matrix.append(row)
            
        decision_matrix = np.array(decision_matrix)

        if expected_constellation is not None and node_to_constellation is not None:
            # 3 Columns: Shape (minimize), Dist (minimize), Affinity (maximize)
            base_weights = np.array([0.3, 0.5, 0.2])
            panic_weights = np.array([0.1, 0.8, 0.1])
            dynamic_weights = base_weights * (1.0 - uncertainty_ratio) + panic_weights * uncertainty_ratio
            weights = dynamic_weights
            criteria_types = ['cost', 'cost', 'benefit']
        else:
            # 2 Columns: Shape (minimize), Dist (minimize)
            weights = np.array([0.5, 0.5])
            criteria_types = ['cost', 'cost']

        if neural_col is not None:
            # 4th column: Siamese embedding distance (minimize). The geometric weights are rescaled by
            # (1 - neural_weight) so the vector still sums to 1 and their relative balance is unchanged.
            weights = np.append(weights * (1.0 - self.neural_weight), self.neural_weight)
            criteria_types = criteria_types + ['cost']

        scores, best_idx = apply_topsis(decision_matrix, weights, criteria_types)
        return self._package(candidates_scores, scores, best_idx)

    def _package(self, candidates_scores, scores, best_idx):
        best_candidate = candidates_scores[best_idx]
        best_score = scores[best_idx]

        # LEVEL 0 diagnostic: runner-up TOPSIS score (hypothesis margin). Read-only —
        # winner selection is untouched.
        _sorted = np.sort(np.asarray(scores, dtype=float))
        score_2nd = float(_sorted[-2]) if len(_sorted) > 1 else float('nan')

        best_route = best_candidate['route']
        matched_map_anchor_id = best_route[0]
        absolute_pos = self.global_map[matched_map_anchor_id]

        return {
            'pos': absolute_pos,
            'route': best_route,
            'score': best_score,
            'score_2nd': score_2nd
        }