"""
sauvc_gym -- a Gymnasium interface to the SAUVC 2026 Stonefish arena.

Quick start
-----------
Terminal 1, the simulator you already have::

    ros2 launch sauvc_stonefish sauvc_qualification.launch.py

Terminal 2::

    python3 -m sauvc_gym.scripts.verify_allocation   # prove the wiring first
    python3 -m sauvc_gym.scripts.random_agent

Registered ids
--------------
``SauvcDepthHold-v0``      one action, depth only. Start here.
``SauvcStationKeeping-v0`` depth + heading + drift. The workhorse.
``SauvcQualGate-v0``       gate transit out and back. Read its docstring first.

All ids take ``vehicle_scn`` as a required kwarg, because the thruster geometry
is read from your scene file rather than duplicated here::

    env = gymnasium.make("SauvcStationKeeping-v0",
                         vehicle_scn="/path/to/my_auv.scn")
"""

from gymnasium.envs.registration import register

from .allocation import AllocationResult, ThrustAllocator
from .envs.auv_base_env import AuvBaseEnv, PoolGeometry
from .envs.gate_env import QualificationGateEnv
from .envs.station_keeping_env import DepthHoldEnv, StationKeepingEnv
from .ros_link import RosLink, VehicleState
from .scn_parse import ThrusterSpec, VehicleSpec, parse_scenario

__version__ = "0.1.0"

__all__ = [
    "AllocationResult", "ThrustAllocator", "AuvBaseEnv", "PoolGeometry",
    "QualificationGateEnv", "DepthHoldEnv", "StationKeepingEnv",
    "RosLink", "VehicleState", "ThrusterSpec", "VehicleSpec", "parse_scenario",
]

register(
    id="SauvcDepthHold-v0",
    entry_point="sauvc_gym.envs.station_keeping_env:DepthHoldEnv",
    max_episode_steps=600,
)
register(
    id="SauvcStationKeeping-v0",
    entry_point="sauvc_gym.envs.station_keeping_env:StationKeepingEnv",
    max_episode_steps=600,
)
register(
    id="SauvcQualGate-v0",
    entry_point="sauvc_gym.envs.gate_env:QualificationGateEnv",
    max_episode_steps=1200,
)
