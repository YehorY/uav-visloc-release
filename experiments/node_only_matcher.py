"""
experiments/node_only_matcher.py — A3b: relocalization WITHOUT graph edges.

The production search builds a video-side landmark ROUTE (a path graph), searches map routes with
A*, and ranks them by edge geometry: internal angles and edge-length RATIOS. This matcher uses only
NODES — landmark positions and classes — to test whether that topological structure carries
localization value.

Hypothesis-and-vote registration, translation only, at the CURRENT nav scale:
  for every video landmark i and every same-class map candidate j:
      hypothesis: video landmark i IS map node j, which fixes the drone translation
          t = map_j - scale * (video_i - video_center)
      project every video landmark k:  m_k = t + scale * (video_k - video_center)
      vote: k matches if a same-class map candidate lies within `match_radius_map_px` of m_k
  best hypothesis = most matches, then lowest mean residual, then enumeration order.
  Bindings: the best hypothesis's matches, one-to-one (greedy by residual), capped at `max_bindings`.

What it deliberately lacks, and why that is the test: edge-length RATIOS are scale-invariant, so the
graph matcher can bind correctly while the scale is still the uncalibrated prior. A node-only matcher
must use absolute positions and therefore depends on the scale being right. At boot the scale is the
0.5 prior (the lock happens during this very commit), so boot is where the value of edges shows first.
"""

import numpy as np


def node_only_match(video_pts, video_cls, map_pts, map_cls, candidate_ids, scale, video_center,
                    match_radius_map_px, min_matches, max_bindings):
    """
    Returns None if no hypothesis reaches `min_matches`, else a dict:
      video_indices, map_indices : parallel binding lists (length <= max_bindings)
      score                      : matches / number of video landmarks, in [0, 1]
      score_2nd                  : the same for the runner-up hypothesis (NaN if none)
      n_hypotheses, n_matches
    """
    V = np.asarray(video_pts, dtype=float).reshape(-1, 2)
    Vc = np.asarray(video_cls)
    cand = np.array(sorted(int(c) for c in candidate_ids), dtype=int)
    if len(V) < 2 or len(cand) == 0:
        return None
    Mp = np.asarray(map_pts, dtype=float)[cand]
    Mc = np.asarray(map_cls)[cand]
    rel = float(scale) * (V - np.asarray(video_center, dtype=float))        # (n, 2) map-px offsets
    same_cls = Vc[:, None] == Mc[None, :]                                    # (n, m)
    r2 = float(match_radius_map_px) ** 2

    results = []   # (n_matches, mean_resid, order, i, j_local)
    order = 0
    for i in range(len(V)):
        for jl in np.nonzero(same_cls[i])[0]:
            t = Mp[jl] - rel[i]
            proj = t + rel                                                   # (n, 2)
            d2 = ((proj[:, None, :] - Mp[None, :, :]) ** 2).sum(-1)          # (n, m)
            d2 = np.where(same_cls, d2, np.inf)
            best = d2.min(axis=1)
            hit = best <= r2
            n_hit = int(hit.sum())
            if n_hit > 0:
                results.append((n_hit, float(np.sqrt(best[hit]).mean()), order, i, int(jl)))
            order += 1
    if not results:
        return None
    results.sort(key=lambda r: (-r[0], r[1], r[2]))
    n_hit, _, _, i, jl = results[0]
    if n_hit < int(min_matches):
        return None

    # Rebuild the winning hypothesis and bind one-to-one by residual.
    t = Mp[jl] - rel[i]
    proj = t + rel
    d = np.sqrt(np.where(same_cls, ((proj[:, None, :] - Mp[None, :, :]) ** 2).sum(-1), np.inf))
    pairs = []
    for k in range(len(V)):
        jk = int(np.argmin(d[k]))
        if d[k, jk] <= float(match_radius_map_px):
            pairs.append((float(d[k, jk]), k, jk))
    pairs.sort()
    used, vid_idx, map_idx = set(), [], []
    for _, k, jk in pairs:
        if jk in used:
            continue
        used.add(jk)
        vid_idx.append(int(k))
        map_idx.append(int(cand[jk]))
        if len(vid_idx) >= int(max_bindings):
            break
    if len(vid_idx) < int(min_matches):
        return None
    n = float(len(V))
    second = next((r for r in results[1:]), None)
    return {"video_indices": vid_idx, "map_indices": map_idx, "score": n_hit / n,
            "score_2nd": (second[0] / n) if second else float("nan"),
            "n_hypotheses": order, "n_matches": n_hit}
