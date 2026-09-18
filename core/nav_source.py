"""
core/nav_source.py
==================
LEVEL 1 — the single owner of navigation parameters the estimator must never derive from pixels.

Heading and scale enter the pipeline ONLY through `NavState`. Before Level 1 they lived in three
places: a module-level heading constant (passed to functions that ignored it, and used only for
radar drawing), the Phase 8.1 flight-controller mock whose leg heading actually drove the
estimator, and a function-local scale with a 0.5 prior and a one-time visual lock.

Hierarchy
  NavState            frozen snapshot: heading_deg, leg_index, leg_changed, scale, scale_locked
  NavSource (ABC)     scale prior + one-time lock; subclasses supply the heading
  StaticNavSource     constant heading (fallback for sequences with no flight-controller data)
  TelemetryNavSource  flight-controller execution-log replay (Phase 8.1 mock logic, verbatim)
  build_nav_source()  factory from the `nav:` and `sequence:` config sections

Scale ordering contract
  The scale is NOT fixed per frame. On the frame where it locks, reads BEFORE the lock (e.g. the
  boot viewport filter) must see the prior, and reads AFTER it (the commit solve, the frame log)
  must see the locked value. `lock_scale()` therefore returns a fresh NavState that the caller
  rebinds; a snapshot taken at the start of the frame would be stale.

Imports: numpy and the standard library only. Nothing here reads video, detections or GT beyond
the waypoint geometry a TelemetryNavSource is constructed with.
"""

import dataclasses
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NavState:
    frame_id: int
    heading_deg: float     # mission-leg heading used by the flashlight, kinematic clamp, 1D gate
    leg_index: int
    leg_changed: bool      # True only on the frame a leg transition is broadcast
    scale: float           # map px per video px
    scale_locked: bool


class NavSource(ABC):
    """Scale prior + one-time lock. Subclasses provide the heading via `_heading()` and must call
    `self.update(0)` at the end of their constructor so `state` is valid before the frame loop."""

    def __init__(self, scale_prior):
        # Stored as given (no numeric cast): a YAML 0.5 is already a Python float, and the locked
        # value keeps whatever type the pipeline computed, so arithmetic stays bit-identical.
        self._scale = scale_prior
        self._scale_locked = False
        self._state = None

    @abstractmethod
    def _heading(self, frame_id):
        """Return (leg_index, heading_deg, leg_changed) for this frame."""

    @abstractmethod
    def describe(self):
        """One-line human-readable description for the run log."""

    def update(self, frame_id):
        leg_index, heading_deg, leg_changed = self._heading(frame_id)
        self._state = NavState(frame_id=int(frame_id), heading_deg=heading_deg,
                               leg_index=int(leg_index), leg_changed=bool(leg_changed),
                               scale=self._scale, scale_locked=self._scale_locked)
        return self._state

    def lock_scale(self, scale):
        """One-time scale lock. Returns the updated NavState, which the caller must rebind."""
        if self._scale_locked:
            raise RuntimeError(f"scale already locked at {self._scale}; refusing to re-lock at {scale}")
        if self._state is None:
            raise RuntimeError("update() must run before lock_scale()")
        self._scale = scale
        self._scale_locked = True
        self._state = dataclasses.replace(self._state, scale=scale, scale_locked=True)
        return self._state

    @property
    def state(self):
        return self._state


class StaticNavSource(NavSource):
    """Constant heading, single leg. The fallback when no flight-controller data exists.

    It is NOT equivalent to a telemetry source on a multi-leg mission: the flashlight, clamp and
    1D gate would all keep using the first leg's direction after every turn."""

    def __init__(self, heading_deg, scale_prior):
        super().__init__(scale_prior)
        self._heading_deg = heading_deg
        self.update(0)

    def _heading(self, frame_id):
        return 0, self._heading_deg, False

    def describe(self):
        return f"StaticNavSource: constant heading {self._heading_deg} deg (no leg changes)."


class TelemetryNavSource(NavSource):
    """
    Flight-controller execution-log replay (Phase 8.1 mock, logic moved verbatim).

    A real flight controller executes waypoint turns from its OWN GPS/INS and broadcasts leg state;
    this replays the recorded passage frames. EPISTEMOLOGICAL BOUNDARY: the waypoint geometry never
    leaves this class — the pipeline receives only NavState scalars. Replace with the real MAVLink
    feed at the X-Plane phase.
    """

    def __init__(self, waypoints_px, passage_frames, scale_prior):
        super().__init__(scale_prior)
        wps = [np.asarray(p, dtype=float) for p in waypoints_px]
        leg_vecs = [wps[i + 1] - wps[i] for i in range(len(wps) - 1)]
        self._leg_headings = [math.degrees(math.atan2(v[1], v[0])) for v in leg_vecs]
        self._passage_frames = sorted(int(f) for f in passage_frames)
        self._n_waypoints = len(wps)
        self._leg = 0
        self.update(0)

    def _heading(self, frame_id):
        leg = sum(1 for f in self._passage_frames if frame_id >= f)
        leg = min(leg, len(self._leg_headings) - 1)
        passed = leg > self._leg
        if passed:
            self._leg = leg
        return self._leg, self._leg_headings[self._leg], passed

    def describe(self):
        return (f"TelemetryNavSource: FC-log replay, {self._n_waypoints} waypoints, "
                f"passages at {self._passage_frames}.")


def build_nav_source(nav_cfg, sequence_cfg, waypoints_px):
    """Factory for the `nav:` config section. `waypoints_px` are the sequence waypoints already
    projected to map pixels (the projection needs map dimensions the pipeline owns)."""
    source = str(nav_cfg["source"]).strip().lower()
    scale_prior = nav_cfg["scale_prior"]
    if source == "telemetry":
        return TelemetryNavSource(waypoints_px, sequence_cfg["waypoint_passage_frames"], scale_prior)
    if source == "static":
        return StaticNavSource(nav_cfg["static_heading_deg"], scale_prior)
    raise ValueError(f"nav.source must be 'telemetry' or 'static'; got {source!r}")
