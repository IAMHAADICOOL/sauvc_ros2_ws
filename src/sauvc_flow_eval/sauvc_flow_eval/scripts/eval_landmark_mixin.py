#!/usr/bin/env python3
"""eval_landmark_mixin.py — landmark localization for flow_eval_node.

Split out of flow_eval_node.py unchanged. Everything that consumes
gate_detector's /vision/features lives here: the anchor-frame reconciliation, the
'gate'/'map' branches, and the 'slam' state-augmentation path. Mixed into
FlowEvalNode, so every `self.` reference resolves exactly as it did before the
split.
"""
import math

import numpy as np

from sauvc_flow_eval.estimators.ekf_estimator import EkfEstimator


class EvalLandmarkMixin:
    """landmark_mode 'gate' | 'map' | 'slam' handling. Mixed into FlowEvalNode."""

    # ---- landmark localization (consumes gate_detector's /vision/features) ----
    def _anchor_x_ned(self):
        """NED x of the state origin (= where dead reckoning starts). FIX(gate frame
        mismatch): the EKF state and the GTSAM graph are START-RELATIVE (both begin at
        0 and only get the GT anchor ADDED at publish time), while gate_x_known is a
        WORLD NED coordinate (4.4 = the gate line, ~16 m ahead of a start at -11.6).
        The old gate branch compared 'gate_x_known - rwx' (a WORLD x) directly against
        ekf.x[0] (a START-RELATIVE x) — off by the whole anchor, so the innovation
        gate (2 m) rejected every real gate observation and the correction could only
        ever fire spuriously. Subtracting the anchor makes measurement and state share
        one frame. Falls back to the parsed scene start before the first GT message
        (and on hardware, where GT never exists)."""
        if self.gt_anchor is not None:
            return float(self.gt_anchor[0])          # compare frame == 'ned' here
        if self.scene_start_ned is not None:
            return float(self.scene_start_ned[0])
        return None
    def on_feature(self, msg):
        try:
            name, bx, by, bz, rng, brg, elev, area = msg.data.split(',')
            bx, by, bz = float(bx), float(by), float(bz)
            rng, brg = float(rng), float(brg)
        except ValueError:
            return
        t = self.get_clock().now().nanoseconds * 1e-9
        anchor_x = self._anchor_x_ned()
        if anchor_x is None:
            return          # world knowledge can't meet a start-relative state yet
        gate_x_state = self.gate_x_known - anchor_x     # rulebook x, state frame
        # body FRD -> NED world (planar; pitch/roll are small at cruise)
        c, s = np.cos(self.yaw_ned), np.sin(self.yaw_ned)
        rwx = c * bx - s * by
        rwy = s * bx + c * by
        obs_var = (self.lm_obs_sigma * max(rng, 1.0)) ** 2
        ex, ey = self.ekf.x[0], self.ekf.x[1]     # current EKF position estimate
        # LIVE PLOT: mark this landmark the moment it is observed, in the EKF
        # colour with a short unique id (F1, F2, ...). Position is anchored to the
        # world so it overlays the trajectory tracks: gate posts use the rulebook x,
        # everything else the EKF robot estimate plus the rotated body observation.
        # In 'slam' mode the SLAM print block later refines the same marker from the
        # settled state estimate (and adds a matching GTSAM-colour marker).
        if self.traj is not None:
            anchor_y = (self.gt_anchor[1] if self.gt_anchor is not None
                        else self.scene_start_ned[1])
            _is_gate_lm = name.startswith('GatePost') or name == 'GateCenter'
            lx = (gate_x_state if _is_gate_lm else ex + rwx) + anchor_x
            ly = (ey + rwy) + anchor_y
            self.traj.add_feature('ekf', name, lx, ly)
        # ---- UPGRADE H: 'slam' -> true state augmentation, then done ----
        if self.lm_mode == 'slam':
            self._on_feature_slam(name, bx, by, bz, rng, brg, gate_x_state, t)
            return
        # ---- GATE: rulebook-known x -> anisotropic absolute correction ----
        if name.startswith('GatePost') or name == 'GateCenter':
            p_meas_x = gate_x_state - rwx          # FIX(gate frame): was gate_x_known
            if abs(p_meas_x - ex) < self.lm_innov_gate:
                self.ekf.update_position_xy(p_meas_x, ey, t, obs_var, 1e12)
                self.gtsam.add_landmark_xy(p_meas_x, ey, np.sqrt(obs_var), 1e6)
                if t - self._lm_log_t > 2.0:
                    self._lm_log_t = t
                    self.get_logger().info(
                        f'GATE x-correction applied: x <- {p_meas_x:+.2f} '
                        f'(innov {p_meas_x - ex:+.2f} m, sigma '
                        f'{np.sqrt(obs_var):.2f})')
            if self.lm_mode != 'map':
                return
            name = 'GateCenter'                    # map the pair as one landmark
        if self.lm_mode != 'map':
            return
        # ---- SMALL MAP: freeze at first quality-gated sightings, then correct ----
        lm = self.lm_map.setdefault(
            name, dict(sum=np.zeros(2), n=0, pos=None, var=None, frozen=False))
        if not lm['frozen']:
            if rng <= self.lm_max_first_range:
                lm['sum'] += np.array([ex + rwx, ey + rwy])
                lm['n'] += 1
                if lm['n'] >= self.lm_min_frames:
                    lm['pos'] = lm['sum'] / lm['n']
                    lm['var'] = obs_var * self.lm_map_inflate
                    lm['frozen'] = True
                    self.get_logger().info(
                        f'MAP: {name} frozen at ({lm["pos"][0]:+.2f}, '
                        f'{lm["pos"][1]:+.2f}) after {lm["n"]} sightings '
                        f'(sigma {np.sqrt(lm["var"]):.2f} m) — absolute error floor '
                        'is the vehicle pose error at THESE sightings.')
            return
        # re-observation of a frozen landmark -> both-axis correction
        p_meas = lm['pos'] - np.array([rwx, rwy])
        innov = np.hypot(p_meas[0] - ex, p_meas[1] - ey)
        if innov < self.lm_innov_gate:
            v = lm['var'] + obs_var
            self.ekf.update_position_xy(p_meas[0], p_meas[1], t, v, v)
            self.gtsam.add_landmark_xy(p_meas[0], p_meas[1], np.sqrt(v), np.sqrt(v))
            if t - self._lm_log_t > 2.0:
                self._lm_log_t = t
                self.get_logger().info(
                    f'MAP re-observation: {name} -> pos <- ({p_meas[0]:+.2f}, '
                    f'{p_meas[1]:+.2f}), innov {innov:.2f} m')
    def _on_feature_slam(self, name, bx, by, bz, rng, brg, gate_x_state, t):
        """UPGRADE H: FEKFSLAM-style state augmentation (EKF) + Point3 landmarks
        (GTSAM). Body FRD observation of a NAMED feature; compare frame is 'ned'
        (enforced in __init__), so body FRD pairs with the NED yaw directly.
        The gate posts' rulebook x enters as a one-shot feature-coordinate
        measurement at birth (state frame, anchor already subtracted); their y is
        LEARNED at birth with the proper cross-covariance and thereafter corrects
        robot y exactly as much as the correlation justifies — the sound version
        of "store gate y, then use it". Everything is guarded: range window +
        N-consistent-sightings birth + chi2 innovation gate live in EkfEstimator;
        median birth + Huber kernels live in GtsamEstimator."""
        # detector noise is natively polar: sigma_r from the size-ranging law
        # (a + b*r^2, see COVARIANCES.md), sigma_brg from the GDT error columns.
        sig_r = self.lm_sig_ra + self.lm_sig_rb * rng * rng
        R_body = EkfEstimator.polar_to_cart_cov(rng, brg, sig_r, self.lm_sig_b)
        is_gate = name.startswith('GatePost') or name == 'GateCenter'
        known = {0: (gate_x_state, self.gate_sigma_x)} if is_gate else None
        status = self.ekf.update_feature(name, bx, by, self.yaw_ned_fused
                                         if self.use_lane else self.yaw_ned,
                                         t, R_body, known=known)
        if status == 'init':
            fe = self.ekf.feature_estimate(name)
            self.get_logger().info(
                f"SLAM: {name} ENTERED THE STATE at "
                f"({fe[0]:+.2f}, {fe[1]:+.2f}) state-frame "
                f"(sigma {math.sqrt(fe[2]):.2f}/{math.sqrt(fe[3]):.2f} m)"
                + (f'; rulebook x={gate_x_state:+.2f} fused '
                   f'(sigma {self.gate_sigma_x})' if is_gate else ''))
        elif status == 'gated' and t - self._lm_log_t > 2.0:
            self._lm_log_t = t
            self.get_logger().warn(
                f'SLAM: {name} observation chi2-GATED '
                f'(rejections so far: {self.ekf.rejected.get(name, 0)}) — '
                'a mis-classified blob or a bumped prop.')
        # GTSAM: birth prior once per gate name (anisotropic: rulebook x tight,
        # y/z huge — the graph-native var_y=1e12), then buffer the observation;
        # it attaches to the next keyframe pose inside add_keyframe.
        if self.gtsam.available and self.gtsam.initialized:
            if is_gate and name not in self._gate_prior_set:
                self.gtsam.set_landmark_prior(
                    name, (gate_x_state, None, None),
                    (self.gate_sigma_x, 1e3, 1e3))
                self._gate_prior_set.add(name)
            self.gtsam.add_landmark_obs(name, (bx, by, bz))
