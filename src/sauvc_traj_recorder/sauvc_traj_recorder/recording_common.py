"""recording_common.py — shared logic for record_trajectory_sim/hw.launch.py.

Both launch files need the same two things: (1) pick the next trajectory_NN folder by
scanning base_dir, and (2) start `ros2 bag record` + the recorder node against it. Only
the TOPIC NAMES differ between sim and hardware, so that's the only thing each launch
file supplies — everything else lives here once instead of being copy-pasted twice.
"""
import os
import re


def resolve_trajectory_dir(base_dir, forced_id):
    """Return (traj_dir, traj_name, already_existed) for this recording run."""
    base_dir = os.path.expanduser(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    forced_id = (forced_id or '').strip()
    if forced_id:
        traj_num = int(forced_id)
    else:
        existing = []
        for name in os.listdir(base_dir):
            m = re.fullmatch(r'trajectory_(\d+)', name)
            if m:
                existing.append(int(m.group(1)))
        traj_num = (max(existing) + 1) if existing else 1

    traj_name = f'trajectory_{traj_num:02d}'
    traj_dir = os.path.join(base_dir, traj_name)
    already_existed = os.path.isdir(traj_dir)
    os.makedirs(traj_dir, exist_ok=True)
    return traj_dir, traj_name, already_existed


def build_recording_actions(traj_dir, traj_name, already_existed, bag_topics,
                            node_params, record_bag, platform_label):
    """Return the list of launch actions: log line, optional bag record, recorder node."""
    from launch.actions import ExecuteProcess, LogInfo
    from launch_ros.actions import Node

    msg = (f"[record_trajectory:{platform_label}] recording {traj_name} -> {traj_dir}\n"
           f"  bag topics: {', '.join(bag_topics) if bag_topics else '(none)'}\n"
           f"  Ctrl+C to stop this take. Relaunch for the next trajectory.")
    if already_existed:
        msg += (f"\n  WARNING: {traj_dir} already exists (trajectory_id was forced). "
                f"ros2 bag record will refuse to write into an existing bag dir and "
                f"exit; the recorder node will overwrite files there and restart "
                f"numbering from 000000.")
    actions = [LogInfo(msg=msg)]

    if record_bag and bag_topics:
        actions.append(ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-o', os.path.join(traj_dir, 'bag'), *bag_topics],
            output='screen'))

    actions.append(Node(
        package='sauvc_traj_recorder', executable='trajectory_recorder_node',
        name='trajectory_recorder_node', output='screen',
        parameters=[node_params]))

    return actions
