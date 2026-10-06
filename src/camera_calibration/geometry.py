"""Rigid transforms use metres, radians and column vectors."""
import itertools
import math
import cv2
import numpy as np


def transform(value):
    t = np.asarray(value, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError("Expected a finite 4x4 transform")
    r = t[:3, :3]
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8) or not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1, atol=1e-5):
        raise ValueError("Invalid rigid transform")
    return t


def inverse(t):
    t = transform(t)
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ t[:3, 3]
    return out


def pose_matrix(pose):
    p = np.asarray(pose, dtype=float)
    if p.shape != (6,) or not np.isfinite(p).all():
        raise ValueError("Expected six finite pose values")
    roll, pitch, yaw = p[3:]
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    t = np.eye(4)
    t[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                 [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]]
    t[:3, 3] = p[:3]
    return t


def sdk_pose(pose_mm_rad):
    p = np.asarray(pose_mm_rad, dtype=float).copy()
    p[:3] /= 1000
    return pose_matrix(p)


def difference(a, b):
    a, b = transform(a), transform(b)
    angle = math.acos(float(np.clip((np.trace(a[:3, :3].T @ b[:3, :3])-1)/2, -1, 1)))
    return float(np.linalg.norm(a[:3, 3]-b[:3, 3])), angle


def mean_transform(ts):
    out = np.eye(4)
    u, _, vt = np.linalg.svd(sum(t[:3, :3] for t in ts))
    out[:3, :3] = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    out[:3, 3] = np.mean([t[:3, 3] for t in ts], axis=0)
    return out


def check_rotation_coverage(poses):
    vectors = []
    for a, b in itertools.combinations(poses, 2):
        v = cv2.Rodrigues(a[:3, :3].T @ b[:3, :3])[0].ravel()
        if np.linalg.norm(v) >= math.radians(5):
            vectors.append(v)
    if len(vectors) < 3:
        raise ValueError("Insufficient rotation: tilt the board about different axes")
    values = np.linalg.svd(np.asarray(vectors), compute_uv=False)
    if values[1] / values[0] < .1:
        raise ValueError("Rotations are nearly parallel; collect tilts about another axis")
    return values.tolist()
