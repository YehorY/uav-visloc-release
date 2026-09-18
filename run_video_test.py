import cv2
import numpy as np
import torch
import json
import math
import os
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO
from core import pathfinder
from core.ensemble_navigator import EnsembleNavigator
from tools.map_extractor import GlobalMapExtractor

from tools.coordinate_toolkit import CoordinateToolkit
from tools.cartographic_registry import CartographicRegistry
from tools.navigation_memory import NavigationMemory
# LEVEL 0 — measurement infrastructure (logging / timing / determinism); no core logic inside.
from core.diagnostics import (FrameLogger, StageProfiler, set_global_determinism,
                              closest_point_on_polyline)
# LEVEL 1 — heading and scale enter the pipeline ONLY through a NavSource (see core/nav_source.py).
from core.nav_source import build_nav_source
# LEVEL 2.3 — ablation code, quarantined in experiments/ and inert unless `experiments:` enables it.
from experiments.dropout import apply_detection_dropout
from experiments.node_only_matcher import node_only_match

# Configuration

# LEVEL 1 — navigation, sensor, sequence and sequence-tuned constants that used to live here are
# now in configs/demo.yaml (sections `sensor`, `nav`, `sequence`, `sequence_specific_tuning`).
# Removed as dead: the module heading constant (only ever passed to parameters nothing read, plus
# radar drawing, which now uses the NavState heading), MAX_PIXEL_SPEED_PER_FRAME,
# MAX_FLIGHT_RADIUS, SECTOR_WIDTH (never read).

def filter_by_camera_viewport(map_nodes_array, current_pos, scale, camera_width_px, camera_height_px):
    """
    Filters map nodes based on an Axis-Aligned Bounding Box (AABB) 
    representing the unrotated camera footprint.
    """
    valid_indices = set()
    
    # Calculate physical width/height of the camera on the map (with 50% margin for safety)
    map_fov_w = camera_width_px * scale * 1.5
    map_fov_h = camera_height_px * scale * 1.5
    
    for i, node in enumerate(map_nodes_array):
        # 1. Vector from drone to node
        dx = node[0] - current_pos[0]
        dy = node[1] - current_pos[1]
        
        # 2. Axis-Aligned Bounds Check (No rotation!)
        in_width = abs(dx) <= (map_fov_w / 2)
        in_height = abs(dy) <= (map_fov_h / 2)
        
        if in_width and in_height:
            valid_indices.add(i)
            
    return valid_indices

def calculate_drone_offset_position(matched_pairs, video_center, scale, inlier_threshold):
    """
    Calculates the drone's global position using Deterministic RANSAC (Exhaustive Consensus).
    Assumes the video feed is globally aligned (Nadir view, North-Up).
    LEVEL 0: returns a 4th value, residual_rms — RMS distance of the accepted vote cluster
    to its mean (NaN when undefined). Diagnostic only; the estimate math is untouched.
    STEP 1 / MODULE 1a: returns a 5th value, inlier_mask — list[bool] aligned with
    matched_pairs, True where that pair belongs to the accepted consensus cluster.
    Diagnostic only (feeds per-track RANSAC-status history); selection logic is untouched.
    """
    if not matched_pairs:
        return None, 0, 0, float('nan'), []

    cw, ch = video_center
    estimated_positions = []
    
    for map_pos, vid_pos in matched_pairs:
        x, y = vid_pos
        
        # Vector from Center of screen to Building in video pixels
        dx = x - cw
        dy = y - ch
        
        # Direct mapping (No rotation needed because video is Axis-Aligned)
        v_map = np.array([dx, dy])
        
        # Project building pos backwards to find drone pos
        drone_pos = np.array(map_pos) - scale * v_map
        estimated_positions.append(drone_pos)
        
    if len(estimated_positions) == 0:
        return None, 0, 0, float('nan'), []
    elif len(estimated_positions) < 3:
        # Not enough points for consensus, return simple mean
        _arr = np.asarray(estimated_positions, dtype=float)
        _rms = float(np.sqrt(np.mean(np.sum((_arr - _arr.mean(axis=0)) ** 2, axis=1))))
        return (np.mean(estimated_positions, axis=0), len(estimated_positions), len(estimated_positions), _rms,
                [True] * len(estimated_positions))
        
    # --- DETERMINISTIC RANSAC (Exhaustive Consensus) ---
    estimated_positions = np.array(estimated_positions)
    
    best_inliers = []
    best_variance = float('inf')
    best_mask = np.zeros(len(estimated_positions), dtype=bool)   # MODULE 1a (diagnostic)
    
    # Evaluate every point as a potential center of truth
    for hypothesis in estimated_positions:
        # Find all points within the threshold of this hypothesis
        distances = np.linalg.norm(estimated_positions - hypothesis, axis=1)
        inliers = estimated_positions[distances <= inlier_threshold]
        
        # Update best consensus
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_mask = distances <= inlier_threshold
            # Calculate variance as a tie-breaker quality metric
            centroid = np.mean(inliers, axis=0)
            best_variance = np.mean(np.sum((inliers - centroid)**2, axis=1))
        elif len(inliers) == len(best_inliers) and len(inliers) > 0:
            # Tie-breaker: choose the tighter, more focused cluster
            centroid = np.mean(inliers, axis=0)
            variance = np.mean(np.sum((inliers - centroid)**2, axis=1))
            if variance < best_variance:
                best_inliers = inliers
                best_variance = variance
                best_mask = distances <= inlier_threshold
                
    # Fallback to median if catastrophic failure (no cluster forms)
    if len(best_inliers) == 0:
        return (np.median(estimated_positions, axis=0), 0, len(estimated_positions), float('nan'),
                [False] * len(estimated_positions))

    # Return the highly confident mean of the consensus cluster
    _cl = np.asarray(best_inliers, dtype=float)
    _rms = float(np.sqrt(np.mean(np.sum((_cl - _cl.mean(axis=0)) ** 2, axis=1))))
    return (np.mean(best_inliers, axis=0), len(best_inliers), len(estimated_positions), _rms,
            [bool(x) for x in best_mask])

def build_constellations(global_anchors, orbit_radius=250.0):
    """
    Groups global anchors into Constellations based on intersecting orbits.
    Returns:
      constellations: list of lists, where each sublist contains the anchor IDs belonging to that constellation.
      node_to_constellation: dict mapping anchor ID to its constellation ID.
    """
    n = len(global_anchors)
    visited = [False] * n
    constellations = []
    node_to_constellation = {}
    
    for i in range(n):
        if not visited[i]:
            # Start a new constellation (BFS)
            current_cluster = []
            queue = [i]
            visited[i] = True
            
            while queue:
                curr = queue.pop(0)
                current_cluster.append(curr)
                node_to_constellation[curr] = len(constellations)
                
                # Check intersections with all other orbits
                for j in range(n):
                    if not visited[j]:
                        dist = np.linalg.norm(global_anchors[curr] - global_anchors[j])
                        if dist <= orbit_radius: # Orbits intersect!
                            visited[j] = True
                            queue.append(j)
                            
            constellations.append(current_cluster)
            
    return constellations, node_to_constellation

def run_video_test(config=None, nav_source=None):
    """Component 4: Integration & Radar Projection in the Main Loop.
    LEVEL 0: accepts a YAML-loaded config dict (see configs/demo.yaml). None loads that file.
    LEVEL 1: `nav_source` injects a NavSource (tests); None builds one from the `nav` section."""
    # ===================== LEVEL 0 — MEASUREMENT INFRASTRUCTURE =====================
    # Config plumbing (defaults preserve legacy standalone behavior), determinism, and
    # the modular Logger/Profiler. NO core algorithmic logic is refactored here.
    if config is None:
        # Standalone `python run_video_test.py`: navigation, sequence and tuning values now live only
        # in the config (no module-level fallbacks), so load the canonical one.
        import yaml
        _default_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "demo.yaml")
        with open(_default_cfg, "r", encoding="utf-8") as _fh:
            config = yaml.safe_load(_fh)
    _data_cfg = config["data"]
    video_path = _data_cfg["video"]
    model_path = _data_cfg["model"]
    map_image_path = _data_cfg["map_image"]
    _run_cfg = config.get("run", {})
    max_frames = int(_run_cfg.get("max_frames", 0)) or None
    write_video = bool(_run_cfg.get("write_video", True))
    # LEVEL 1 — 3-tier configuration. Required keys use [] so a missing value fails loudly
    # instead of silently falling back to different behaviour.
    _sensor_cfg = config["sensor"]
    camera_width_px = _sensor_cfg["camera_width_px"]
    camera_height_px = _sensor_cfg["camera_height_px"]
    map_gsd_m_per_px = _sensor_cfg["map_gsd_m_per_px"]
    _nav_cfg = config["nav"]
    _seq_cfg = config["sequence"]
    gps_waypoints = _seq_cfg["gps_waypoints"]
    _tune_cfg = config["sequence_specific_tuning"]
    clamp_forward_pxf = _tune_cfg["kinematic_clamp_pxf"]["forward"]
    clamp_lateral_pxf = _tune_cfg["kinematic_clamp_pxf"]["lateral"]
    clamp_backward_pxf = _tune_cfg["kinematic_clamp_pxf"]["backward"]
    kinematic_1d_tol_px = _tune_cfg["kinematic_1d_gate"]["tol_px"]
    kinematic_1d_drift_rate = _tune_cfg["kinematic_1d_gate"]["drift_rate_px_per_frame"]
    _gate_cfg = _tune_cfg["gating"]
    gate_min_inlier_ratio = float(_gate_cfg["min_inlier_ratio"])
    gate_min_topsis = float(_gate_cfg["min_topsis_score"])
    detector_landmark_classes = _sensor_cfg["detector_landmark_classes"]
    _trk = _tune_cfg["pixel_tracker"]
    tracking_sector_radius_px = _trk["sector_radius_px"]
    max_missed_frames = _trk["max_missed_frames"]
    velocity_weight_old = _trk["velocity_weight_old"]
    velocity_weight_new = _trk["velocity_weight_new"]
    max_coasting_frames = _tune_cfg["coasting"]["max_frames"]
    coast_velocity_lookback_frames = _tune_cfg["coasting"]["velocity_lookback_frames"]
    scale_lock_min_pair_px = _tune_cfg["scale_lock"]["min_pair_separation_video_px"]
    ransac_inlier_threshold = _tune_cfg["ransac"]["inlier_threshold_map_px"]
    _reloc = _tune_cfg["relocalization"]
    handoff_max_pairs = _reloc["handoff_max_pairs"]
    photo_route_max_vertices = _reloc["photo_route_max_vertices"]
    orbit_radius_map_px = _reloc["orbit_radius_map_px"]
    flashlight_beam_base_deg = _reloc["flashlight"]["beam_base_deg"]
    flashlight_beam_growth_deg = _reloc["flashlight"]["beam_growth_deg"]
    flashlight_radius_base = _reloc["flashlight"]["radius_base_map_px"]
    flashlight_radius_growth = _reloc["flashlight"]["radius_growth_map_px"]
    astar_beam_width = _reloc["a_star"]["beam_width"]
    astar_angle_weight = _reloc["a_star"]["angle_weight"]
    astar_target_percentage = _reloc["a_star"]["target_percentage"]
    # LEVEL 2.3 — ablation toggles. Absent or default = production behaviour (proven cell-identical).
    _exp_cfg = config.get("experiments", {}) or {}
    _drop_cfg = _exp_cfg.get("detection_dropout", {}) or {}
    exp_dropout_rate = float(_drop_cfg.get("rate", 0.0))
    exp_dropout_seed = int(_drop_cfg.get("seed", 0))
    exp_dropout_exempt = set(int(f) for f in _drop_cfg.get("exempt_frames", [1]))
    exp_topsis_criteria = _exp_cfg.get("topsis_criteria", None)
    exp_graph_mode = str(_exp_cfg.get("graph_mode", "route")).strip().lower()
    if exp_graph_mode not in ("route", "nodes_only"):
        raise ValueError(f"experiments.graph_mode must be route or nodes_only; got {exp_graph_mode!r}")
    _node_cfg = _exp_cfg.get("node_only", {}) or {}
    exp_node_radius = float(_node_cfg.get("match_radius_map_px", 35.0))
    exp_node_min_matches = int(_node_cfg.get("min_matches", 3))
    exp_exclude_coasted = bool(_exp_cfg.get("exclude_coasted_votes", False))
    _out_cfg = config.get("output", {})
    set_global_determinism(int(config.get("seed", 42)))
    frame_logger = FrameLogger(_out_cfg.get("frame_log", "out/frame_log.csv"))
    # LEVEL 3 — optional per-frame track log for the offline demo renderer. Read-only with respect to
    # the pipeline; off unless output.track_log is set. Frame log is byte-identical either way.
    _track_log_path = _out_cfg.get("track_log")
    track_log_f = None
    if _track_log_path:
        os.makedirs(os.path.dirname(_track_log_path) or ".", exist_ok=True)
        track_log_f = open(_track_log_path, "w", encoding="utf-8")
    profiler = StageProfiler()
    print(f"[LEVEL0] Gating: min_inlier_ratio={gate_min_inlier_ratio}, "
          f"min_topsis_score={gate_min_topsis}; max_frames={max_frames or 'all'}")
    print(f"[LEVEL1] Sequence '{_seq_cfg['id']}': nav.source={_nav_cfg['source']}, "
          f"scale_prior={_nav_cfg['scale_prior']}, camera={camera_width_px}x{camera_height_px}, "
          f"gsd={map_gsd_m_per_px} m/px")
    if (exp_dropout_rate > 0.0 or exp_topsis_criteria or exp_graph_mode != "route" or exp_exclude_coasted):
        print(f"[ABLATION] dropout={exp_dropout_rate} (seed {exp_dropout_seed}, exempt {sorted(exp_dropout_exempt)}) "
              f"topsis_criteria={exp_topsis_criteria} graph_mode={exp_graph_mode} "
              f"exclude_coasted_votes={exp_exclude_coasted}")
    # ================================================================================

    print(f"Initializing YOLO model from {model_path}...")
    model = YOLO(model_path)
    # LEVEL 0 — device allocation: bind the detector to CUDA when available (t_yolo was
    # 90.1% of runtime on CPU). Falls back to CPU transparently on GPU-less hosts.
    yolo_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(yolo_device)
    print(f"[INIT] YOLO inference device: {yolo_device}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    
    global_anchors_array, global_classes_array = GlobalMapExtractor.extract_anchors(map_image_path, model_path)
    if len(global_anchors_array) == 0:
        print("Error: No anchors found on the global map. Cannot initialize navigator.")
        return

    print("Initializing EnsembleNavigator...")
    navigator = EnsembleNavigator(
        global_anchors_array=global_anchors_array,
        global_classes_array=global_classes_array
    )

    # Initialize Topological Constellations
    constellations, node_to_constellation = build_constellations(global_anchors_array, orbit_radius=orbit_radius_map_px)
    print(f"[INIT] Formed {len(constellations)} Constellations from {len(global_anchors_array)} global anchors.")
    
    # Generate unique colors for constellations (for radar visualization)
    np.random.seed(42)
    constellation_colors = [tuple(int(x) for x in np.random.randint(50, 255, 3)) for _ in range(len(constellations))]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open {video_path}. Ensure the file exists.")
        return

    # Read first frame to get dimensions for VideoWriter
    ret, test_frame = cap.read()
    if not ret:
        print("Failed to read the first frame of the video.")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # reset to start

    # Initialize Advanced Architecture Components
    toolkit = CoordinateToolkit(gsd=map_gsd_m_per_px)
    registry = CartographicRegistry(merge_threshold=7.5) # Based on relative spatial distance (meters)
    memory = NavigationMemory(frame_trigger=90, dist_trigger=50.0)

    # Radar Configuration
    radar_size = 600
    
    # Calculate concatenated video dimensions
    h, w = test_frame.shape[:2]
    # We will resize Scanner View so its height matches radar_size (600)
    aspect_ratio = w / h
    resized_w = int(radar_size * aspect_ratio)
    combined_w = resized_w + radar_size
    combined_h = radar_size
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # LEVEL 0: run.write_video=false skips MP4 encoding (fast path). Overlay drawing still
    # executes, so no pipeline state can differ; only the disk encode is removed.
    out_video = (cv2.VideoWriter('output_slam.mp4', fourcc, 30.0, (combined_w, combined_h))
                 if write_video else None)
    radar_base = np.zeros((radar_size, radar_size, 3), dtype=np.uint8)
    
    # Calculate GPS bounds for consistent projection
    lats = [wp[0] for wp in gps_waypoints]
    lons = [wp[1] for wp in gps_waypoints]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    
    # Need 4K map dimensions for accurate initial state
    map_img = cv2.imread(map_image_path)   # LEVEL 1: was a module constant that ignored data.map_image
    if map_img is not None:
        map_h, map_w = map_img.shape[:2]
    else:
        map_h, map_w = 2000, 4000 # fallback

    # Ground Truth path in 4K Map Pixels
    map_points = [toolkit.gps_to_map_pixel(lat, lon, min_lat, max_lat, min_lon, max_lon, map_w, map_h) 
                  for lat, lon in gps_waypoints]
    
    # Ground Truth path on radar
    radar_points = [toolkit.gps_to_radar_pixel(lat, lon, min_lat, max_lat, min_lon, max_lon, radar_size) 
                    for lat, lon in gps_waypoints]
    
    for i in range(len(radar_points) - 1):
        cv2.line(radar_base, radar_points[i], radar_points[i+1], (255, 0, 0), 2) # Blue GT
    for pt in radar_points:
        cv2.circle(radar_base, pt, 5, (255, 0, 0), -1)

    # Initial State tracked in Global Map (4K) Pixels
    drone_current_pos = np.array([map_points[0][0], map_points[0][1]], dtype=float)

    # LEVEL 1 — NAVIGATION SOURCE. Heading and scale enter the pipeline ONLY through NavState;
    # nothing below derives either from pixels. An injected source (tests) takes precedence.
    if nav_source is None:
        nav_source = build_nav_source(_nav_cfg, _seq_cfg, map_points)
    nav = nav_source.state
    print(f"[NAV] {nav_source.describe()} Frame-0 heading = {nav.heading_deg:.1f} deg; "
          f"scale prior {nav.scale} (unlocked).")
    drone_trajectory_history = [drone_current_pos.copy()]
    last_velocity = np.array([0.0, 0.0])
    prev_centroids_pixel = []
    current_candidate_ids = set()
    
    frame_count = 0
    successful_tracking_frames = 0
    inlier_ratios = []
    frame_times = []
    ttff_frames = None

    print("Starting SLAM-like processing loop. Press 'q' to quit.")

    tracked_map_bindings = {} 
    global_correction_offset = np.array([0.0, 0.0]) # Neutralizes handoff drift
    is_slam_initialized = False # Tracks the very first lock
    lost_tracking_frames = 0
    predicted_pos = drone_current_pos.copy()

    # PHASE 8.1 PORT 2 state — panic-only 1D kinematic gate
    last_valid_along = None           # along-leg projection of the last GENUINE fix
    last_genuine_fix_frame = 0        # blind-time clock for the dynamic tolerance

    # Video Pixel Tracking Memory
    # Format: { anchor_id: {'pos': np.array([x,y]), 'vel': np.array([vx,vy]), 'cls': class_id, 'missed_frames': 0} }
    video_trackers = {}
    next_anchor_id = 1
    # STEP 1 / MODULE 1a — track bookkeeping (telemetry only; tracker behavior unchanged)
    TRACK_HIST_M = 5          # ring-buffer length for conf / area / residual / inlier history
    JPX_MIN_STREAK = 5        # consecutive hits required before a track contributes to J_px
    ID_SWITCH_PX = 25.0       # transform-prior residual above this -> suspected ID switch
    consensus_hist = []       # [(frame, raw_pos)] from continuation solves; cleared on commits

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start_time = time.time()

        frame_count += 1
        if max_frames is not None and frame_count > max_frames:
            print(f"[LEVEL0] max_frames={max_frames} reached — stopping (config-bounded demo run).")
            break

        # LEVEL 0: deterministic capture timestamp (from the video stream, not wall clock)
        timestamp_capture = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        profiler.new_frame(frame_count)
        diag = frame_logger.new_row(frame_count)
        # LEVEL 3 track-log bookkeeping for this frame (read-only telemetry)
        _tl_solve_ids, _tl_solve_mask = [], []
        _tl_search = _tl_commit = _tl_reject = False
        diag["timestamp_capture"] = timestamp_capture

        # LEVEL 1: advance the nav source. Heading steps with the leg; the scale is carried as
        # prior or locked. `nav` is REBOUND wherever the scale locks (see lock_scale below).
        nav = nav_source.update(frame_count)
        if nav.leg_changed:
            print(f"[NAV] Waypoint {nav.leg_index} passage broadcast -> "
                  f"leg {nav.leg_index}, heading {nav.heading_deg:.1f} deg.")

        # 1. Extract raw centroids from YOLO
        with profiler.stage("t_yolo"):
            results = model(frame, verbose=False, device=yolo_device)
        raw_centroids = []
        raw_classes = []
        raw_confs = []      # MODULE 1a: detector confidence per detection (telemetry only)
        raw_bboxes = []     # MODULE 1a: (w, h) per detection (telemetry only)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in detector_landmark_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    raw_centroids.append((cx, cy))
                    raw_classes.append(cls_id)
                    raw_confs.append(float(box.conf[0]))
                    raw_bboxes.append((float(x2 - x1), float(y2 - y1)))
                    
                    # Draw raw YOLO boxes
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 100, 0), 1)

        # LEVEL 2.3 / A4: random detection dropout (sparse-landmark regime). Exempt frames keep every
        # detection so the boot scale lock is not what gets tested.
        if exp_dropout_rate > 0.0 and frame_count not in exp_dropout_exempt:
            raw_centroids, raw_classes, raw_confs, raw_bboxes = apply_detection_dropout(
                raw_centroids, raw_classes, raw_confs, raw_bboxes,
                exp_dropout_rate, exp_dropout_seed, frame_count)

        # 2. Global Map Registry & Spatial NMS (Deduplication)
        # active_anchors = registry.register_or_update(raw_centroids, raw_classes, toolkit, frame_count)
        
        # === PREDICTIVE PIXEL TRACKER ===
        _t_trk0 = time.perf_counter()          # t_track stage (see add_stage_ms below)
        active_anchors = []
        unmatched_raw = list(range(len(raw_centroids)))
        
        # 1. Predict new positions for existing trackers
        for t_id, t_data in list(video_trackers.items()):
            expected_box_pos = t_data['pos'] + t_data['vel']
            best_match_idx = -1
            min_dist = tracking_sector_radius_px
            
            # 2. Find the best raw YOLO detection inside the Expected Sector
            for i in unmatched_raw:
                if raw_classes[i] == t_data['cls']:
                    dist = np.linalg.norm(np.array(raw_centroids[i]) - expected_box_pos)
                    if dist < min_dist:
                        min_dist = dist
                        best_match_idx = i
            
            if best_match_idx != -1:
                # MATCH FOUND! Update coordinates, calculate new velocity, reset missed frames
                new_pos = np.array(raw_centroids[best_match_idx])
                # Simple momentum for velocity: 80% old, 20% new
                t_data['vel'] = t_data['vel'] * velocity_weight_old + (new_pos - t_data['pos']) * velocity_weight_new
                t_data['pos'] = new_pos
                t_data['missed_frames'] = 0
                # MODULE 1a bookkeeping — read by telemetry only; pos/vel/cls/missed untouched
                _bw, _bh = raw_bboxes[best_match_idx]
                t_data['pos_raw'] = new_pos
                t_data['bbox_wh'] = (_bw, _bh)
                t_data['age'] = t_data.get('age', 0) + 1
                t_data['hits'] = t_data.get('hits', 0) + 1
                t_data['hit_streak'] = t_data.get('hit_streak', 0) + 1
                t_data['last_hit_frame'] = frame_count
                t_data['source'] = 'yolo'
                for _k, _v in (('conf_hist', raw_confs[best_match_idx]), ('area_hist', _bw * _bh),
                               ('resid_hist', float(min_dist)), ('raw_hist', new_pos.copy())):
                    _h = t_data.setdefault(_k, [])
                    _h.append(_v)
                    if len(_h) > (3 if _k == 'raw_hist' else TRACK_HIST_M):
                        del _h[0]
                
                active_anchors.append({'id': t_id, 'pos_pixel': new_pos, 'cls': t_data['cls']})
                unmatched_raw.remove(best_match_idx)
            else:
                # TRACK LOST (Flicker). Coast by moving it along its velocity vector.
                t_data['missed_frames'] += 1
                t_data['hit_streak'] = 0      # MODULE 1a: continuity broken
                t_data['age'] = t_data.get('age', 0) + 1
                t_data['raw_hist'] = []       # J_px needs 3 CONSECUTIVE raw detections
                t_data['source'] = 'coast'
                if t_data['missed_frames'] < max_missed_frames:
                    t_data['pos'] = expected_box_pos # Update position blindly
                    active_anchors.append({'id': t_id, 'pos_pixel': expected_box_pos, 'cls': t_data['cls']})
                else:
                    # Dead track. Remove it.
                    del video_trackers[t_id]

        # 3. Register brand new tracks for remaining unmatched YOLO detections
        for i in unmatched_raw:
            new_pos = np.array(raw_centroids[i])
            video_trackers[next_anchor_id] = {
                'pos': new_pos, 
                'vel': np.array([0.0, 0.0]), 
                'cls': raw_classes[i], 
                'missed_frames': 0,
                # MODULE 1a bookkeeping (telemetry only)
                'pos_raw': new_pos, 'bbox_wh': raw_bboxes[i], 'age': 1, 'hits': 1, 'hit_streak': 1,
                'last_hit_frame': frame_count, 'source': 'yolo',
                'conf_hist': [raw_confs[i]], 'area_hist': [raw_bboxes[i][0] * raw_bboxes[i][1]],
                'resid_hist': [], 'raw_hist': [new_pos.copy()], 'inlier_hist': []
            }
            active_anchors.append({'id': next_anchor_id, 'pos_pixel': new_pos, 'cls': raw_classes[i]})
            next_anchor_id += 1
        profiler.add_stage_ms("t_track", (time.perf_counter() - _t_trk0) * 1000.0)
        
        # Draw Deduplicated Anchors and Semantic Constellation
        n_anchors = len(active_anchors)
        for i in range(n_anchors):
            pt1 = (int(active_anchors[i]['pos_pixel'][0]), int(active_anchors[i]['pos_pixel'][1]))
            obj_id = active_anchors[i]['id']
            
            # Highlight active global anchor
            cv2.circle(frame, pt1, 6, (0, 0, 255), -1)
            cv2.putText(frame, f"ID:{obj_id}", (pt1[0] + 10, pt1[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
            # Draw Constellation lines ONLY between currently visible deduplicated anchors
            for j in range(i + 1, n_anchors):
                pt2 = (int(active_anchors[j]['pos_pixel'][0]), int(active_anchors[j]['pos_pixel'][1]))
                dist = np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
                if dist < 200:
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 1)

        matched_pairs_this_frame = []
        matched_ids_this_frame = []   # MODULE 1a: track id per pair (inlier-mask history)
        for anchor in active_anchors:
            anchor_id = anchor['id']
            if anchor_id in tracked_map_bindings:
                matched_pairs_this_frame.append((tracked_map_bindings[anchor_id], anchor['pos_pixel']))
                matched_ids_this_frame.append(anchor_id)
        # LEVEL 2.3 / A_coast: the continuation SOLVE may exclude coasted tracks. Only the solve changes;
        # the handoff trigger and the scale lock still see every bound pair (single variable).
        if exp_exclude_coasted:
            _keep = [k for k, _tid in enumerate(matched_ids_this_frame)
                     if video_trackers.get(_tid, {}).get('last_hit_frame') == frame_count]
            solve_pairs = [matched_pairs_this_frame[k] for k in _keep]
            solve_ids = [matched_ids_this_frame[k] for k in _keep]
        else:
            solve_pairs, solve_ids = matched_pairs_this_frame, matched_ids_this_frame
        
        video_center = (w / 2, h / 2)
        is_tracking_valid = False

        # MODULE 1a — ID-SWITCH PROXY (telemetry only). For each bound track matched by YOLO
        # THIS frame, predict its video position from its map binding plus the RANSAC consensus
        # extrapolated from the two previous consecutive continuation solves:
        #   raw_pred = 2*raw[t-1] - raw[t-2];   vid_pred = center + (map_pt - raw_pred) / scale
        # A large residual means the detection sits away from where its map binding says the
        # landmark should be — the signature of a track that latched onto a different object.
        if (nav.scale_locked and len(consensus_hist) >= 2
                and consensus_hist[-1][0] == frame_count - 1 and consensus_hist[-2][0] == frame_count - 2):
            _raw_pred = 2.0 * consensus_hist[-1][1] - consensus_hist[-2][1]
            _res = []
            for _tid, _mp in tracked_map_bindings.items():
                _tr = video_trackers.get(_tid)
                if _tr is None or _tr.get('last_hit_frame') != frame_count:
                    continue
                _vp = np.array(video_center, dtype=float) + (np.asarray(_mp, dtype=float) - _raw_pred) / nav.scale
                _res.append(float(np.linalg.norm(np.asarray(_tr['pos_raw'], dtype=float) - _vp)))
            if _res:
                diag["n_bound_hits"] = len(_res)
                diag["id_switch_proxy"] = sum(1 for _r in _res if _r > ID_SWITCH_PX)
                diag["prior_resid_med"] = float(np.median(_res))

        if len(matched_pairs_this_frame) >= 2:
            # --- ONE-TIME SCALE CALIBRATION (STRICT) ---
            if not nav.scale_locked and len(matched_pairs_this_frame) >= 2:
                max_vid_dist = 0
                best_pair = None
                for i in range(len(matched_pairs_this_frame)):
                    for j in range(i + 1, len(matched_pairs_this_frame)):
                        map_pt1, vid_pt1 = matched_pairs_this_frame[i]
                        map_pt2, vid_pt2 = matched_pairs_this_frame[j]
                        
                        dist_vid = np.linalg.norm(np.array(vid_pt1) - np.array(vid_pt2))
                        if dist_vid > max_vid_dist:
                            max_vid_dist = dist_vid
                            best_pair = (map_pt1, vid_pt1, map_pt2, vid_pt2)
                            
                if best_pair and max_vid_dist > scale_lock_min_pair_px:
                    map_pt1, vid_pt1, map_pt2, vid_pt2 = best_pair
                    dist_map = np.linalg.norm(np.array(map_pt1) - np.array(map_pt2))
                    # LEVEL 1: one-time lock; REBIND nav so every later read this frame sees it
                    nav = nav_source.lock_scale(dist_map / max_vid_dist)
                    print(f"[SLAM] Strict Scale locked at: {nav.scale:.4f}")
                    # [SCALE-TELEM] READ-ONLY diagnostic (Step 0.1) — no logic change.
                    _sids = [a['id'] for a in active_anchors
                             if np.array_equal(a['pos_pixel'], vid_pt1) or np.array_equal(a['pos_pixel'], vid_pt2)]
                    print(f"[SCALE-TELEM] pair_ids={_sids} map_dist={dist_map:.1f}px vid_dist={max_vid_dist:.1f}px "
                          f"vid=({vid_pt1[0]:.0f},{vid_pt1[1]:.0f})<->({vid_pt2[0]:.0f},{vid_pt2[1]:.0f}) "
                          f"map=({map_pt1[0]:.0f},{map_pt1[1]:.0f})<->({map_pt2[0]:.0f},{map_pt2[1]:.0f}) "
                          f"n_pairs={len(matched_pairs_this_frame)}")
            # ---------------------------------------------

            with profiler.stage("t_ransac"):
                raw_calculated_pos, inliers_cnt, total_pairs, residual_rms, inlier_mask = calculate_drone_offset_position(
                    matched_pairs=solve_pairs,
                    video_center=video_center, scale=nav.scale,
                    inlier_threshold=ransac_inlier_threshold
                )

            if raw_calculated_pos is not None:
                _ir = (inliers_cnt / total_pairs) if total_pairs > 0 else 0.0
                _tl_solve_ids, _tl_solve_mask = list(solve_ids), list(inlier_mask)   # LEVEL 3 telemetry
                # MODULE 1a: per-track RANSAC-status history + consensus history (telemetry only)
                for _tid, _in in zip(solve_ids, inlier_mask):
                    if _tid in video_trackers:
                        _h = video_trackers[_tid].setdefault('inlier_hist', [])
                        _h.append(bool(_in))
                        if len(_h) > TRACK_HIST_M:
                            del _h[0]
                consensus_hist.append((frame_count, np.asarray(raw_calculated_pos, dtype=float).copy()))
                if len(consensus_hist) > 3:
                    del consensus_hist[0]
                if total_pairs > 0:
                    inlier_ratios.append(_ir)
                # LEVEL 0 diagnostics for this solve
                diag["n_inliers"] = int(inliers_cnt)
                diag["inlier_ratio"] = float(_ir)
                diag["residual_rms"] = float(residual_rms)

                # ===== LEVEL 0 — CONFIDENCE GATING (checked BEFORE the state update) =====
                # Below-threshold solve: retain the previous estimate, flag the frame as
                # low_confidence, and DO NOT publish a new position. Tracking state itself is
                # untouched (no core state-machine refactor at Level 0).
                if total_pairs > 0 and _ir < gate_min_inlier_ratio:
                    diag["low_confidence"] = 1
                    predicted_pos = drone_current_pos
                    is_tracking_valid = True
                    lost_tracking_frames = 0
                else:
                    with profiler.stage("t_transform_fit"):
                        # 1. APPLY HARD LOCK OFFSET
                        target_pos = raw_calculated_pos + global_correction_offset

                        # 2. PHASE 8.2 STRATEGY B — ASYMMETRIC KINEMATIC CLAMP (leg-frame).
                        # Replaces the isotropic 40px/f cap. Innovations within the caps land exactly
                        # on target (identical to old behavior for small deltas); oversized transients
                        # are clamped per-component instead of chased at full speed.
                        delta = target_pos - drone_current_pos
                        _cth = math.radians(nav.heading_deg)
                        _fwd = np.array([math.cos(_cth), math.sin(_cth)])
                        _lat = np.array([-_fwd[1], _fwd[0]])
                        d_along = float(np.dot(delta, _fwd))
                        d_cross = float(np.dot(delta, _lat))
                        if d_along >= 0.0:
                            d_along = min(d_along, clamp_forward_pxf)
                        else:
                            d_along = max(d_along, -clamp_backward_pxf)
                        _da_pre, _dc_pre = d_along, d_cross     # clamp-activity telemetry
                        d_cross = max(-clamp_lateral_pxf, min(clamp_lateral_pxf, d_cross))
                        diag["clamp_hit"] = int(d_along != _da_pre or d_cross != _dc_pre)
                        drone_current_pos = drone_current_pos + d_along * _fwd + d_cross * _lat

                    predicted_pos = drone_current_pos
                    is_tracking_valid = True
                    lost_tracking_frames = 0 # Reset coasting timer
                    # PHASE 8.1: refresh the 1D gate reference from this GENUINE fix
                    _gth = math.radians(nav.heading_deg)
                    last_valid_along = (drone_current_pos[0] * math.cos(_gth)
                                        + drone_current_pos[1] * math.sin(_gth))
                    last_genuine_fix_frame = frame_count

        if not is_tracking_valid and is_slam_initialized:
            # We lost visual lock. Should we coast or panic?
            if lost_tracking_frames < max_coasting_frames:
                # COASTING MODE
                lost_tracking_frames += 1
                
                # Calculate clean Macro-Velocity
                coast_vel = np.array([0.0, 0.0])
                history_len = len(drone_trajectory_history)
                if history_len > 1:
                    lookback = min(coast_velocity_lookback_frames, history_len - 1)
                    coast_vel = (drone_current_pos - drone_trajectory_history[-lookback - 1]) / lookback
                    
                drone_current_pos = drone_current_pos + coast_vel
                predicted_pos = drone_current_pos
                is_tracking_valid = True # Pretend we are still tracking to suppress A*
                print(f"[SLAM] Coasting with Macro-Velocity... (Frame {lost_tracking_frames}/{max_coasting_frames})")
            else:
                # Timer expired. We are officially lost.
                is_tracking_valid = False

        # --- SLAM: Event-Driven Pipeline (Relocalization & Seamless Handoff) ---
        trigger_search = False
        if not is_slam_initialized:
            trigger_search = True # First Boot
        elif not is_tracking_valid:
            trigger_search = True # Panic Mode
        elif is_tracking_valid and len(matched_pairs_this_frame) <= handoff_max_pairs:
            trigger_search = True # Pre-emptive Handoff Warning

        if trigger_search and len(active_anchors) >= 2:
            # PHASE 8.1: the 1D gate applies ONLY to panic re-locks — a pre-emptive handoff
            # (healthy tracking) compares across the offset-absorption frame gap and must pass.
            panic_search = (not is_tracking_valid) and is_slam_initialized
            _tl_search = True   # LEVEL 3 telemetry: a relocalization search runs this frame
            if is_tracking_valid and is_slam_initialized:
                print(f"\n[Frame {frame_count}] Low Anchor Warning (<=2). Triggering Seamless Handoff...")
            else:
                print(f"\n[Frame {frame_count}] Tracking Lost or First Boot! Triggering Topological Search...")
            
            # 1. Build Local Graph
            ref_photo_nodes_array = np.array([a['pos_pixel'] for a in active_anchors])
            ref_photo_classes_array = np.array([a.get('cls', a.get('class_id')) for a in active_anchors])
            ref_photo_anchor_id = 0
            
            # --- PREDICTIVE FLASHLIGHT (LOOK-AHEAD) ---
            expected_constellation = None
            
            # 1. Base uncertainty metric
            uncertainty_ratio = min(1.0, lost_tracking_frames / max_coasting_frames)
            
            # 2. Central beam angle — the nav source's CURRENT leg (Phase 8.1); at boot this is
            # leg 0 (-126.8 deg vs the old constant 225 = -135 deg: the ~8 deg shift covered by
            # the empirical identity contract on scale 0.6266 / pair [11,24]).
            theta_center = math.radians(nav.heading_deg)
            
            # 3. Dynamic Beam Width
            current_beam_width = math.radians(flashlight_beam_base_deg) + (math.radians(flashlight_beam_growth_deg) * uncertainty_ratio)
            
            # 4. Dynamic Search Radius
            current_radius = flashlight_radius_base + (flashlight_radius_growth * uncertainty_ratio)
            
            # Gaussian Constellation Voting
            constellation_votes = {}
            for idx, pt in enumerate(global_anchors_array):
                dx = pt[0] - drone_current_pos[0]
                dy = pt[1] - drone_current_pos[1]
                distance = math.hypot(dx, dy)
                
                if distance <= current_radius:
                    angle = math.atan2(dy, dx)
                    angular_diff = angle - theta_center
                    # Normalize to -pi to pi
                    angular_diff = (angular_diff + math.pi) % (2 * math.pi) - math.pi
                    
                    if abs(angular_diff) <= current_beam_width:
                        weight = math.exp(-0.5 * (angular_diff / (current_beam_width / 2.0))**2)
                        c_id = node_to_constellation.get(idx)
                        if c_id is not None:
                            constellation_votes[c_id] = constellation_votes.get(c_id, 0.0) + weight
                            
            if constellation_votes:
                expected_constellation = max(constellation_votes, key=constellation_votes.get)
                print(f"   -> [PREDICT] Flashlight expects Constellation ID: {expected_constellation}")
            # ------------------------------------------

            with profiler.stage("t_graph_build"):
                if exp_graph_mode == "nodes_only":
                    # LEVEL 2.3 / A3b: no video-side graph. Every landmark is a node; the matcher below
                    # replaces this list with the landmarks it actually binds.
                    ref_photo_route = list(range(len(ref_photo_nodes_array)))
                else:
                    ref_photo_route = pathfinder.find_photo_path_beam_search(
                        photo_nodes_array=ref_photo_nodes_array,
                        ref_photo_classes_array=ref_photo_classes_array,
                        anchor_id=ref_photo_anchor_id,
                        max_vertices=min(photo_route_max_vertices, len(ref_photo_nodes_array))
                    )

            # LEVEL 0 diagnostics: photo-graph structure (path graph: E = V - 1)
            if ref_photo_route:
                diag["n_graph_nodes"] = len(ref_photo_route)
                diag["n_graph_edges"] = 0 if exp_graph_mode == "nodes_only" else max(0, len(ref_photo_route) - 1)

            if ref_photo_route and len(ref_photo_route) >= 2:
                with profiler.stage("t_candidate_gen"):
                    # 2. Rectangular Viewport Pre-filter
                    candidate_node_ids = filter_by_camera_viewport(
                        map_nodes_array=global_anchors_array,
                        current_pos=drone_current_pos,
                        scale=nav.scale,
                        camera_width_px=camera_width_px,
                        camera_height_px=camera_height_px
                    )
                    current_candidate_ids = candidate_node_ids

                    all_map_routes = []
                    _node_match = None
                    if exp_graph_mode == "nodes_only":
                        # LEVEL 2.3 / A3b: node-only hypothesis voting inside the same viewport, at the
                        # current nav scale. One binding set replaces the A* route candidates.
                        _node_match = node_only_match(
                            ref_photo_nodes_array, ref_photo_classes_array,
                            global_anchors_array, global_classes_array, candidate_node_ids,
                            nav.scale, video_center, exp_node_radius, exp_node_min_matches,
                            photo_route_max_vertices)
                        if _node_match is not None:
                            ref_photo_route = _node_match["video_indices"]
                            all_map_routes = [_node_match["map_indices"]]
                    elif len(candidate_node_ids) >= len(ref_photo_route):
                        # 3. A* Expansion within the filtered sector
                        for map_anchor_id in candidate_node_ids:
                            map_routes = pathfinder.find_map_paths_A_star_beam_search(
                                map_nodes_array=global_anchors_array,
                                map_classes_array=global_classes_array,
                                anchor_id=map_anchor_id,
                                candidate_node_ids=candidate_node_ids,
                                ref_photo_nodes_array=ref_photo_nodes_array,
                                ref_photo_classes_array=ref_photo_classes_array,
                                ref_photo_route=ref_photo_route,
                                beam_width=astar_beam_width,
                                angle_weight=astar_angle_weight,
                                target_percentage=astar_target_percentage
                            )
                            all_map_routes.extend(map_routes)
                diag["n_candidates"] = len(all_map_routes)

                # 4. Ensemble Selection
                if all_map_routes:
                    with profiler.stage("t_topsis"):
                        if _node_match is not None:
                            best_candidate = {'route': _node_match["map_indices"], 'score': _node_match["score"],
                                              'score_2nd': _node_match["score_2nd"]}
                        else:
                            best_candidate = navigator.evaluate_candidates(
                                ref_photo_nodes_array=ref_photo_nodes_array,
                                ref_photo_route=ref_photo_route,
                                all_map_routes=all_map_routes,
                                expected_constellation=expected_constellation,
                                node_to_constellation=node_to_constellation,
                                uncertainty_ratio=uncertainty_ratio,
                                criteria=exp_topsis_criteria
                            )

                    if best_candidate is not None:
                        diag["topsis_score_best"] = float(best_candidate['score'])
                        diag["topsis_score_2nd"] = float(best_candidate.get('score_2nd', float('nan')))

                    # ===== LEVEL 0 — CONFIDENCE GATING (relocalization commit) =====
                    # A commit whose TOPSIS score is below threshold is not published: the
                    # previous estimate is retained and the frame is flagged low_confidence.
                    if (best_candidate is not None
                            and float(best_candidate['score']) < gate_min_topsis):
                        print(f"[LEVEL0-GATE] Commit blocked: topsis_score "
                              f"{best_candidate['score']:.3f} < {gate_min_topsis} — "
                              f"retaining previous estimate (low_confidence).")
                        diag["low_confidence"] = 1
                        best_candidate = None

                    if best_candidate is not None:
                        print("[SLAM] Match Found! Locking new landmarks...")
                        _tl_commit = True   # LEVEL 3 telemetry
                        tracked_map_bindings.clear()

                        global_route_indices = best_candidate['route']
                        local_route_indices = ref_photo_route 

                        # Lock the new bindings
                        matched_pairs_this_frame = []
                        commit_ids = []   # MODULE 1a: track id per pair (inlier-mask history)
                        for i in range(min(len(global_route_indices), len(local_route_indices))):
                            map_pt = global_anchors_array[global_route_indices[i]]
                            vid_anchor_id = active_anchors[local_route_indices[i]]['id']
                            vid_pt = active_anchors[local_route_indices[i]]['pos_pixel']

                            tracked_map_bindings[vid_anchor_id] = map_pt
                            matched_pairs_this_frame.append((map_pt, vid_pt))
                            commit_ids.append(vid_anchor_id)

                        # --- ONE-TIME SCALE CALIBRATION (STRICT) ---
                        if not nav.scale_locked and len(matched_pairs_this_frame) >= 2:
                            max_vid_dist = 0
                            best_pair = None
                            for i in range(len(matched_pairs_this_frame)):
                                for j in range(i + 1, len(matched_pairs_this_frame)):
                                    map_pt1, vid_pt1 = matched_pairs_this_frame[i]
                                    map_pt2, vid_pt2 = matched_pairs_this_frame[j]
                                    
                                    dist_vid = np.linalg.norm(np.array(vid_pt1) - np.array(vid_pt2))
                                    if dist_vid > max_vid_dist:
                                        max_vid_dist = dist_vid
                                        best_pair = (map_pt1, vid_pt1, map_pt2, vid_pt2)
                                        
                            if best_pair and max_vid_dist > scale_lock_min_pair_px:
                                map_pt1, vid_pt1, map_pt2, vid_pt2 = best_pair
                                dist_map = np.linalg.norm(np.array(map_pt1) - np.array(map_pt2))
                                # LEVEL 1: one-time lock; REBIND nav so every later read this frame sees it
                                nav = nav_source.lock_scale(dist_map / max_vid_dist)
                                print(f"[SLAM] Strict Scale locked at: {nav.scale:.4f}")
                                # [SCALE-TELEM] READ-ONLY diagnostic (Step 0.1) — no logic change.
                                _sids = [a['id'] for a in active_anchors
                                         if np.array_equal(a['pos_pixel'], vid_pt1) or np.array_equal(a['pos_pixel'], vid_pt2)]
                                print(f"[SCALE-TELEM] pair_ids={_sids} map_dist={dist_map:.1f}px vid_dist={max_vid_dist:.1f}px "
                                      f"vid=({vid_pt1[0]:.0f},{vid_pt1[1]:.0f})<->({vid_pt2[0]:.0f},{vid_pt2[1]:.0f}) "
                                      f"map=({map_pt1[0]:.0f},{map_pt1[1]:.0f})<->({map_pt2[0]:.0f},{map_pt2[1]:.0f}) "
                                      f"n_pairs={len(matched_pairs_this_frame)}")
                        # ---------------------------------------------

                        with profiler.stage("t_ransac"):
                            raw_calculated_pos, inliers_cnt, total_pairs, residual_rms, inlier_mask = calculate_drone_offset_position(
                                matched_pairs=matched_pairs_this_frame,
                                video_center=video_center, scale=nav.scale,
                                inlier_threshold=ransac_inlier_threshold
                            )
                        if raw_calculated_pos is not None:
                            # MODULE 1a: per-track RANSAC-status history (telemetry only)
                            _tl_solve_ids, _tl_solve_mask = list(commit_ids), list(inlier_mask)   # LEVEL 3 telemetry
                            for _tid, _in in zip(commit_ids, inlier_mask):
                                if _tid in video_trackers:
                                    _h = video_trackers[_tid].setdefault('inlier_hist', [])
                                    _h.append(bool(_in))
                                    if len(_h) > TRACK_HIST_M:
                                        del _h[0]
                            diag["n_inliers"] = int(inliers_cnt)
                            diag["inlier_ratio"] = float(inliers_cnt / total_pairs) if total_pairs > 0 else 0.0
                            diag["residual_rms"] = float(residual_rms)

                        # === PHASE 8.1 PORT 2: PANIC-ONLY DYNAMIC 1D KINEMATIC GATE ===
                        # Boot-inert by double guard (last_valid_along None + panic_search False).
                        # Pre-emptive handoffs NEVER gated (offset-absorption frame mismatch).
                        # Self-healing on false refusal: reference frozen, truth advances ~10px/f,
                        # tau grows 4px/f -> a refused TRUE re-lock is admitted within frames.
                        kin1d_reject = False
                        if (raw_calculated_pos is not None and last_valid_along is not None
                                and panic_search):
                            _blind = max(0, frame_count - last_genuine_fix_frame)
                            if _blind >= 1:
                                _kth = math.radians(nav.heading_deg)
                                _along_new = (raw_calculated_pos[0] * math.cos(_kth)
                                              + raw_calculated_pos[1] * math.sin(_kth))
                                _tol = kinematic_1d_tol_px + kinematic_1d_drift_rate * _blind
                                if _along_new < last_valid_along - _tol:
                                    kin1d_reject = True
                                    _tl_reject = True   # LEVEL 3 telemetry
                                    print(f"[KINEMATIC-1D] REJECT (panic): dAlong = "
                                          f"{_along_new - last_valid_along:+.0f}px vs tol {_tol:.0f}px "
                                          f"@ {_blind} blind frames — backward re-lock refused.")
                                    tracked_map_bindings.clear()
                                    matched_pairs_this_frame = []
                                    consensus_hist.clear()   # MODULE 1a

                        if raw_calculated_pos is not None and not kin1d_reject:
                            if total_pairs > 0:
                                inlier_ratios.append(inliers_cnt / total_pairs)
                            # Whether FIRST BOOT or HANDOFF, we NEVER overwrite the physical drone_current_pos.
                            # We absorb the visual-to-physical discrepancy into the offset.
                            global_correction_offset = drone_current_pos - raw_calculated_pos
                            consensus_hist.clear()   # MODULE 1a: binding set changed; raw not comparable
                            if not is_slam_initialized:
                                ttff_frames = frame_count
                            is_slam_initialized = True
                            
                            predicted_pos = drone_current_pos
                            is_tracking_valid = True
                            lost_tracking_frames = 0 # Reset coasting timer upon successful handoff
                    else:
                        print("   -> EnsembleNavigator rejected all candidates.")
                else:
                    print("   -> No paths generated in the NPC Sector.")

        # LEVEL 2: per-frame tracking state for coverage-honest metrics. Mirrors exactly what the
        # run's own metrics count: LSR counts state 1; the ATE trajectory history appends states 1 and 2.
        diag["track_state"] = (1 if lost_tracking_frames == 0 else 2) if is_tracking_valid else 0
        if is_tracking_valid:
            drone_trajectory_history.append(drone_current_pos.copy())
            if lost_tracking_frames == 0: successful_tracking_frames += 1
            # [KIN-TELEM] READ-ONLY diagnostic (Step 0.1) — per-frame estimate position.
            print(f"[KIN-TELEM] f={frame_count} x={drone_current_pos[0]:.1f} y={drone_current_pos[1]:.1f}")

        # 3. Store step in short_term_buffer
        memory.add_step(drone_current_pos, active_anchors)

        # 5. Projection on Global Radar
        radar_frame = radar_base.copy()
        # Map drone_current_pos from 4K Map pixels down to radar_size pixels
        radar_pred_x = int(predicted_pos[0] * (radar_size / map_w))
        radar_pred_y = int(predicted_pos[1] * (radar_size / map_h))
        
        # Draw Topological Constellations (Background)
        radar_orbit_radius = int(orbit_radius_map_px * (radar_size / max(map_w, map_h)))
        
        for c_id, members in enumerate(constellations):
            color = constellation_colors[c_id]
            
            # 1. Draw Intracluster Edges (The skeleton of the constellation)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pt1 = global_anchors_array[members[i]]
                    pt2 = global_anchors_array[members[j]]
                    if np.linalg.norm(pt1 - pt2) <= orbit_radius_map_px * 1.5: # Connect close neighbors visually
                        rx1, ry1 = int(pt1[0] * (radar_size / map_w)), int(pt1[1] * (radar_size / map_h))
                        rx2, ry2 = int(pt2[0] * (radar_size / map_w)), int(pt2[1] * (radar_size / map_h))
                        cv2.line(radar_frame, (rx1, ry1), (rx2, ry2), color, 1)
            
            # 2. Draw Orbits and Stars
            for node_id in members:
                node = global_anchors_array[node_id]
                rx = int(node[0] * (radar_size / map_w))
                ry = int(node[1] * (radar_size / map_h))
                
                # Faint orbit
                cv2.circle(radar_frame, (rx, ry), radar_orbit_radius, tuple(c//3 for c in color), 1)
                # The Star
                cv2.circle(radar_frame, (rx, ry), 3, color, -1)
            
        # Draw the Filtered Sector Candidates (Foreground)
        for node_id in current_candidate_ids:
            node = global_anchors_array[node_id]
            rx = int(node[0] * (radar_size / map_w))
            ry = int(node[1] * (radar_size / map_h))
            cv2.circle(radar_frame, (rx, ry), 4, (255, 255, 0), -1)

        # Draw Active Tracking Tethers
        if is_tracking_valid and matched_pairs_this_frame:
            for map_pos, _ in matched_pairs_this_frame:
                anchor_rx = int(map_pos[0] * (radar_size / map_w))
                anchor_ry = int(map_pos[1] * (radar_size / map_h))
                # Draw thin tether from predicted drone pos to active anchor
                cv2.line(radar_frame, (radar_pred_x, radar_pred_y), (anchor_rx, anchor_ry), (200, 200, 200), 1)

        # --- Draw Predictive Flashlight (Phased Array Radar Visual) ---
        uncertainty_ratio_radar = min(1.0, lost_tracking_frames / max_coasting_frames)
        theta_center = math.radians(nav.heading_deg)   # LEVEL 1 (D1): draw the estimator's heading
        
        # 1. Vastly expand map-scale radii to make it visible on radar
        current_beam_width = math.radians(15) + (math.radians(60) * uncertainty_ratio_radar)
        current_radius_map = 800.0 + (1200.0 * uncertainty_ratio_radar) # Up to 2000 map pixels
        radar_radius = int(current_radius_map * (radar_size / max(map_w, map_h)))
        
        angle_start = math.degrees(theta_center - current_beam_width)
        angle_end = math.degrees(theta_center + current_beam_width)
        
        # 2. Create Glowing Transparent Overlay
        overlay = radar_frame.copy()
        cv2.ellipse(overlay, (radar_pred_x, radar_pred_y), (radar_radius, radar_radius), 
                    0, angle_start, angle_end, (0, 150, 255), -1) # Filled Cone
        cv2.addWeighted(overlay, 0.4, radar_frame, 0.6, 0, radar_frame) # 40% Opacity
        
        # 3. Draw Solid Edges
        cv2.ellipse(radar_frame, (radar_pred_x, radar_pred_y), (radar_radius, radar_radius), 0, angle_start, angle_end, (0, 200, 255), 2)
        pt1_x = int(radar_pred_x + radar_radius * math.cos(theta_center - current_beam_width))
        pt1_y = int(radar_pred_y + radar_radius * math.sin(theta_center - current_beam_width))
        pt2_x = int(radar_pred_x + radar_radius * math.cos(theta_center + current_beam_width))
        pt2_y = int(radar_pred_y + radar_radius * math.sin(theta_center + current_beam_width))
        cv2.line(radar_frame, (radar_pred_x, radar_pred_y), (pt1_x, pt1_y), (0, 200, 255), 2)
        cv2.line(radar_frame, (radar_pred_x, radar_pred_y), (pt2_x, pt2_y), (0, 200, 255), 2)

        # Draw the Drone Trajectory History (Green Line)
        for i in range(1, len(drone_trajectory_history)):
            pt1_x = int(drone_trajectory_history[i-1][0] * (radar_size / map_w))
            pt1_y = int(drone_trajectory_history[i-1][1] * (radar_size / map_h))
            pt2_x = int(drone_trajectory_history[i][0] * (radar_size / map_w))
            pt2_y = int(drone_trajectory_history[i][1] * (radar_size / map_h))
            cv2.line(radar_frame, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 255, 0), 2)

        # Draw the Heading Vector (The "Flashlight")
        heading_rad = math.radians(nav.heading_deg)    # LEVEL 1 (D1): draw the estimator's heading
        line_length = 80 # pixels on radar
        end_x = int(radar_pred_x + line_length * math.cos(heading_rad))
        end_y = int(radar_pred_y + line_length * math.sin(heading_rad))
        cv2.line(radar_frame, (radar_pred_x, radar_pred_y), (end_x, end_y), (0, 0, 255), 2)
        
        # Draw Camera FOV Footprint (Axis-Aligned)
        map_fov_w = camera_width_px * nav.scale
        map_fov_h = camera_height_px * nav.scale
        
        # Local unrotated corners
        local_corners = [
            (-map_fov_w / 2, -map_fov_h / 2),
            (map_fov_w / 2, -map_fov_h / 2),
            (map_fov_w / 2, map_fov_h / 2),
            (-map_fov_w / 2, map_fov_h / 2)
        ]
        
        global_corners = []
        for lx, ly in local_corners:
            # Simple translation (no rotation)
            gx = predicted_pos[0] + lx
            gy = predicted_pos[1] + ly
            
            # Scale to radar size using the existing scaling logic
            rx = int(gx * (radar_size / map_w))
            ry = int(gy * (radar_size / map_h))
            global_corners.append([rx, ry])
            
        cv2.polylines(radar_frame, [np.array(global_corners)], isClosed=True, color=(255, 0, 255), thickness=1)
        # --------------------------------
        
        cv2.circle(radar_frame, (radar_pred_x, radar_pred_y), 6, (0, 0, 255), -1)

        # Resize Scanner View and concatenate
        frame_resized = cv2.resize(frame, (resized_w, combined_h))
        combined_frame = cv2.hconcat([frame_resized, radar_frame])
        if out_video is not None:
            out_video.write(combined_frame)
        
        frame_times.append(time.time() - start_time)

        # ===================== LEVEL 0 — PER-FRAME RECORD =====================
        diag["n_detections"] = len(raw_centroids)
        # MODULE 1a telemetry: detector confidence, persistence, raw-centroid jitter J_px.
        # J_px = RMS pixel second difference |p_t - 2p_{t-1} + p_{t-2}| over tracks with
        # >= JPX_MIN_STREAK consecutive hits: constant motion cancels, leaving jitter.
        if raw_confs:
            diag["mean_det_conf"] = float(np.mean(raw_confs))
        diag["n_confirmed"] = sum(1 for _t in video_trackers.values() if _t.get('hit_streak', 0) >= JPX_MIN_STREAK)
        _j2 = []
        for _t in video_trackers.values():
            _rh = _t.get('raw_hist', [])
            if _t.get('hit_streak', 0) >= JPX_MIN_STREAK and len(_rh) == 3:
                _d = np.asarray(_rh[2], dtype=float) - 2.0 * np.asarray(_rh[1], dtype=float) + np.asarray(_rh[0], dtype=float)
                _j2.append(float(_d[0] ** 2 + _d[1] ** 2))
        if _j2:
            diag["J_px"] = float(np.sqrt(np.mean(_j2)))
            diag["n_jpx_tracks"] = len(_j2)
        if len(active_anchors) >= 2:
            _pts = np.array([a['pos_pixel'] for a in active_anchors], dtype=float)
            diag["landmark_spread_x"] = float(np.std(_pts[:, 0]))
            diag["landmark_spread_y"] = float(np.std(_pts[:, 1]))
        diag["est_x"] = float(drone_current_pos[0])
        diag["est_y"] = float(drone_current_pos[1])
        _gt = closest_point_on_polyline(drone_current_pos, map_points)
        diag["gt_x"], diag["gt_y"] = float(_gt[0]), float(_gt[1])
        diag["scale"] = float(nav.scale)
        diag["heading_used"] = float(nav.heading_deg)
        if track_log_f is not None:
            # LEVEL 3: track roles for the demo renderer. Precedence: coasting (not detected this frame)
            # > bound_inlier / bound_outlier (RANSAC mask of this frame's final solve; on commit frames the
            # commit solve) > bound_unsolved (bound, but no solve ran) > unbound.
            _mask = dict(zip(_tl_solve_ids, _tl_solve_mask))
            _tracks = []
            for _a in active_anchors:
                _tid = _a['id']
                _tr = video_trackers.get(_tid, {})
                _bound = _tid in tracked_map_bindings
                if _tr.get('last_hit_frame') != frame_count:
                    _role = "coasting"
                elif _bound and _tid in _mask:
                    _role = "bound_inlier" if _mask[_tid] else "bound_outlier"
                elif _bound:
                    _role = "bound_unsolved"
                else:
                    _role = "unbound"
                _mp = tracked_map_bindings.get(_tid)
                _wh = _tr.get('bbox_wh', (0.0, 0.0))
                _tracks.append({"id": int(_tid), "x": round(float(_a['pos_pixel'][0]), 2),
                                "y": round(float(_a['pos_pixel'][1]), 2), "w": round(float(_wh[0]), 1),
                                "h": round(float(_wh[1]), 1), "cls": int(_a['cls']), "role": _role,
                                "map": ([round(float(_mp[0]), 1), round(float(_mp[1]), 1)] if _mp is not None else None)})
            _dets = [[round(float(_c[0]), 1), round(float(_c[1]), 1), round(float(_b[0]), 1), round(float(_b[1]), 1), int(_k)]
                     for _c, _b, _k in zip(raw_centroids, raw_bboxes, raw_classes)]
            track_log_f.write(json.dumps({"f": frame_count, "search": _tl_search, "commit": _tl_commit,
                                          "reject": _tl_reject, "dets": _dets, "tracks": _tracks}) + "\n")
        frame_logger.log(diag)
        profiler.end_frame(frame_times[-1] * 1000.0)
        # ======================================================================
        print(f"Rendering frame {frame_count}...", end="\r")

    print("\nProcessing complete! Generating SLAM Metrics Report...")
    cap.release()
    if out_video is not None:
        out_video.release()

    # ===================== LEVEL 0 — END-OF-RUN OUTPUTS =====================
    frame_logger.close()
    if track_log_f is not None:
        track_log_f.close()
        print(f"[LEVEL3] Track log written: {_track_log_path}")
    profiler.save_csv(_out_cfg.get("profile_log", "out/profile.csv"))
    print(profiler.distribution_table())
    # ========================================================================

    # --- METRICS CALCULATION ---
    # 1. Localization Success Rate (LSR)
    lsr = (successful_tracking_frames / frame_count) * 100 if frame_count > 0 else 0

    # 2. Absolute Trajectory Error (Cross-Track Error to Ground Truth Waypoints)
    def point_to_segment_dist(pt, v1, v2):
        line_vec = v2 - v1
        pt_vec = pt - v1
        line_len = np.linalg.norm(line_vec)
        if line_len == 0: return np.linalg.norm(pt_vec)
        proj = np.dot(pt_vec, line_vec / line_len)
        if proj < 0: return np.linalg.norm(pt_vec)
        if proj > line_len: return np.linalg.norm(pt - v2)
        return np.linalg.norm(pt_vec - (proj * (line_vec / line_len)))

    errors = []
    for pt in drone_trajectory_history:
        # Find distance to the closest GT segment
        min_dist = float('inf')
        for i in range(len(map_points) - 1):
            p1 = np.array(map_points[i])
            p2 = np.array(map_points[i+1])
            dist = point_to_segment_dist(np.array(pt), p1, p2)
            if dist < min_dist:
                min_dist = dist
        errors.append(min_dist)
    
    rmse_ate = np.sqrt(np.mean(np.square(errors)))

    # Calculate New Metrics
    gsd_meters = map_gsd_m_per_px
    rmse_ate_meters = rmse_ate * gsd_meters
    avg_inlier_ratio = (np.mean(inlier_ratios) * 100) if inlier_ratios else 0.0
    avg_fps = 1.0 / np.mean(frame_times) if frame_times else 0.0
    
    # Calculate Local Trajectory Jitter (Frame-to-Frame displacement)
    frame_deltas = []
    for i in range(1, len(drone_trajectory_history)):
        dist = np.linalg.norm(drone_trajectory_history[i] - drone_trajectory_history[i-1])
        frame_deltas.append(dist * gsd_meters)
    jitter_std = np.std(frame_deltas) if frame_deltas else 0.0

    print(f"\n=== UAV-VisLoc PERFORMANCE METRICS ===")
    print(f"Total Frames Processed: {frame_count}")
    print(f"Algorithm Speed: {avg_fps:.1f} FPS")
    print(f"Time To First Fix (TTFF): {ttff_frames if ttff_frames else 'Failed'} frames")
    print(f"Localization Success Rate (LSR): {lsr:.2f}%")
    print(f"Average RANSAC Inlier Ratio: {avg_inlier_ratio:.1f}%")
    print(f"Local Trajectory Jitter (StdDev): {jitter_std:.2f} meters/frame")
    print(f"Absolute Trajectory Error (RMSE): {rmse_ate_meters:.2f} meters")
    print(f"======================================\n")

    # --- PLOT GENERATION ---
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot([p[0] for p in drone_trajectory_history], [p[1] for p in drone_trajectory_history], label='UAV-VisLoc Trajectory', color='green', linewidth=2)
        plt.plot([p[0] for p in map_points], [p[1] for p in map_points], label='GPS Ground Truth', color='blue', linestyle='--', linewidth=2)
        
        plt.title('UAV-VisLoc vs Ground Truth Trajectory')
        plt.xlabel('Global Map X (Pixels)')
        plt.ylabel('Global Map Y (Pixels)')
        plt.gca().invert_yaxis() # Image coordinates (Y goes down)
        
        # Add metrics to plot
        metrics_text = (
            f"ATE: {rmse_ate_meters:.2f} m\n"
            f"LSR: {lsr:.2f}%\n"
            f"FPS: {avg_fps:.1f}\n"
            f"Inliers: {avg_inlier_ratio:.1f}%\n"
            f"Jitter: {jitter_std:.2f} m/f\n"
            f"TTFF: {ttff_frames if ttff_frames else 'N/A'}"
        )
        plt.text(0.02, 0.02, metrics_text, transform=plt.gca().transAxes, 
                 fontsize=10, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.legend(loc='upper right')
        plt.grid(True)
        
        report_path = 'slam_metrics_report.png'
        plt.savefig(report_path)
        print(f"Metrics plot saved to {report_path}")
    except ImportError:
        print("Matplotlib not found. Skipping plot generation.")

if __name__ == "__main__":
    run_video_test()
