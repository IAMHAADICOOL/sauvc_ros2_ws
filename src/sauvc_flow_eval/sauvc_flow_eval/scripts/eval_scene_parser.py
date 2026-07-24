#!/usr/bin/env python3
"""eval_scene_parser.py — Stonefish .scn parsing for flow_eval_node.

Split out of flow_eval_node.py unchanged. `re` and `xml.etree.ElementTree` stay
imported INSIDE the function exactly as they were, so a run that never passes
scene_file still pays nothing for them.
"""


def _parse_scene(path):
    """Parse a Stonefish .scn: vehicle start pose + landmark world positions.
    Returns (start_xyz_ned | None, start_yaw, {name: (x, y, z)}). Landmarks are every
    <static> or <dynamic> whose name matches gate/flare/drum/tub, plus a derived
    'GateCenter' (midpoint of GatePostPort/GatePostStbd - the map point whose x is the
    rulebook-known gate line). $(find ...) substitutions only appear in file paths, so
    the attributes needed here parse as literal XML.
    """
    import re
    import xml.etree.ElementTree as ET
    root = ET.fromstring(open(path).read())
    start_xyz, start_yaw = None, 0.0
    for inc in root.iter('include'):
        f = inc.get('file', '')
        if 'my_auv' in f or 'vehicle' in f:
            for arg in inc.iter('arg'):
                if arg.get('name') == 'start_position':
                    start_xyz = tuple(float(v) for v in arg.get('value').split())
                elif arg.get('name') == 'start_yaw':
                    start_yaw = float(arg.get('value'))
    pat = re.compile(r'gate|flare|drum|tub', re.I)
    landmarks = {}
    for tag in ('static', 'dynamic'):
        for el in root.iter(tag):
            name = el.get('name', '')
            wt = el.find('world_transform')
            if name and pat.search(name) and wt is not None:
                landmarks[name] = tuple(float(v) for v in wt.get('xyz').split())
    if 'GatePostPort' in landmarks and 'GatePostStbd' in landmarks:
        p_, s_ = landmarks['GatePostPort'], landmarks['GatePostStbd']
        landmarks['GateCenter'] = tuple((a + b) / 2 for a, b in zip(p_, s_))
    return start_xyz, start_yaw, landmarks
