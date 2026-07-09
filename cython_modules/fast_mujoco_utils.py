from __future__ import annotations

import numpy as np


def extract_data(model, data, joint_names=None):
    """Pure-Python fallback with the same role as the Cython helper in the annex."""
    joint_names = joint_names or []
    qpos = np.zeros(len(joint_names), dtype=float)
    qvel = np.zeros(len(joint_names), dtype=float)
    if model is None or data is None:
        return qpos, qvel
    for i, name in enumerate(joint_names):
        try:
            joint_id = model.joint(name).id
            qpos_addr = model.jnt_qposadr[joint_id]
            qvel_addr = model.jnt_dofadr[joint_id]
            qpos[i] = data.qpos[qpos_addr]
            qvel[i] = data.qvel[qvel_addr]
        except Exception:
            continue
    return qpos, qvel
