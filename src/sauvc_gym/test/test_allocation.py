"""Unit tests for the parts that do not need a simulator.

Run: python3 -m pytest test/ -v
These import no rclpy, on purpose -- the maths must be testable on any machine.
"""
import numpy as np
import pytest

from sauvc_gym.allocation import ThrustAllocator
from sauvc_gym.scn_parse import parse_scenario, rpy_to_matrix

SCN = "test/fixtures/vehicle_example.scn"


@pytest.fixture
def alloc():
    return ThrustAllocator(parse_scenario(SCN))


def test_parses_eight_thrusters():
    spec = parse_scenario(SCN)
    assert spec.n_thrusters == 8
    assert spec.names == ["HFP", "HFS", "HAP", "HAS", "VFP", "VFS", "VAP", "VAS"]
    assert spec.setpoint_topic == "/sauvc_auv/thruster_setpoints"


def test_max_thrust_matches_validated_value():
    # Kt=0.0005, w_max=314 -> ~49 N. This is the empirically confirmed figure;
    # if it drifts, the scene file or the parser changed.
    spec = parse_scenario(SCN)
    assert spec.thrusters[0].max_thrust == pytest.approx(49.3, abs=0.2)


def test_setpoint_law_is_quadratic_not_linear():
    """Half thrust needs sqrt(0.5)=0.707 of setpoint, not 0.5.

    This is the single easiest thing to get wrong in the whole package.
    """
    t = parse_scenario(SCN).thrusters[0]
    assert t.thrust_from_setpoint(0.5) == pytest.approx(0.25 * t.max_thrust)
    assert t.setpoint_from_thrust(0.5 * t.max_thrust) == pytest.approx(0.7071, abs=1e-3)


def test_setpoint_thrust_round_trip():
    t = parse_scenario(SCN).thrusters[0]
    for u in np.linspace(-1, 1, 21):
        assert t.setpoint_from_thrust(t.thrust_from_setpoint(u)) == pytest.approx(u, abs=1e-9)


def test_left_handed_prop_with_inverted_setpoint_cancels():
    spec = parse_scenario(SCN)
    by_name = {t.name: t for t in spec.thrusters}
    # HFS is right="false" + inverted_setpoint="true" -> the two cancel
    assert by_name["HFS"].right_handed is False
    assert by_name["HFS"].inverted_setpoint is True
    assert by_name["HFS"].setpoint_sign == pytest.approx(1.0)
    # HFP is right="true", no inversion -> also +1
    assert by_name["HFP"].setpoint_sign == pytest.approx(1.0)


def test_lone_inversion_flips_sign():
    """Exactly one of (left-handed, inverted) must give -1.

    This is the 'vehicle doesn't move then topples' failure mode.
    """
    from sauvc_gym.scn_parse import ThrusterSpec
    t = ThrusterSpec("X", np.zeros(3), np.array([1.0, 0, 0]), 0.0005, 314.0,
                     right_handed=False, inverted_setpoint=False)
    assert t.setpoint_sign == -1.0
    t2 = ThrusterSpec("X", np.zeros(3), np.array([1.0, 0, 0]), 0.0005, 314.0,
                      right_handed=True, inverted_setpoint=True)
    assert t2.setpoint_sign == -1.0


def test_groups_split_by_axis_not_by_name(alloc):
    assert alloc._horizontal.sum() == 4
    assert alloc._vertical.sum() == 4


def test_pure_commands_do_not_leak(alloc):
    for k, dof in enumerate(alloc.action_dofs):
        a = np.zeros(4)
        a[k] = 1.0
        w = alloc.allocate(a).wrench_delivered
        for j in range(6):
            from sauvc_gym.allocation import DOF_NAMES
            if DOF_NAMES[j] != dof:
                assert abs(w[j]) < 1e-6, f"{dof} leaked into {DOF_NAMES[j]}"


def test_saturation_preserves_direction(alloc):
    """Over-command everything; the horizontal wrench direction must survive."""
    r = alloc.allocate(np.array([1.0, 1.0, 0.0, 1.0]))
    req, got = r.wrench_requested, r.wrench_delivered
    rows = [0, 1, 5]  # Fx, Fy, Mz -- the horizontal group
    ratios = np.array([got[j] / req[j] for j in rows])
    assert np.allclose(ratios, ratios[0], atol=1e-9), "direction distorted"
    assert r.saturation < 1.0


def test_setpoints_always_legal(alloc):
    rng = np.random.default_rng(1)
    for _ in range(500):
        r = alloc.allocate(rng.uniform(-1, 1, 4))
        assert np.all(np.abs(r.setpoints) <= 1.0 + 1e-9)


def test_forward_inverse_consistent(alloc):
    rng = np.random.default_rng(2)
    for _ in range(200):
        r = alloc.allocate(rng.uniform(-1, 1, 4))
        assert np.allclose(alloc.setpoints_to_wrench(r.setpoints),
                           r.wrench_delivered, atol=1e-9)


def test_rpy_matrix_is_orthonormal():
    r = rpy_to_matrix(0.3, -0.7, 1.9)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_v_floor_profile():
    """1.6 m at the centreline, 1.2 m at the end walls."""
    from sauvc_gym.envs.auv_base_env import PoolGeometry
    pool = PoolGeometry(use_floor_profile=True)
    assert pool.floor_depth(0.0) == pytest.approx(1.6)
    assert pool.floor_depth(12.5) == pytest.approx(1.2)
    assert pool.floor_depth(-12.5) == pytest.approx(1.2)
    flat = PoolGeometry(use_floor_profile=False)
    assert flat.floor_depth(12.5) == pytest.approx(2.0)


def test_bad_dof_name_rejected():
    with pytest.raises(ValueError, match="unknown DOFs"):
        ThrustAllocator(parse_scenario(SCN), action_dofs=("surge", "wiggle"))


def test_arena_scn_gives_useful_error(tmp_path):
    p = tmp_path / "arena.scn"
    p.write_text('<?xml version="1.0"?><scenario><include file="v.scn"/></scenario>')
    with pytest.raises(ValueError, match="no <robot> element"):
        parse_scenario(str(p))
