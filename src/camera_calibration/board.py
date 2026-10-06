"""ChArUco and GridBoard detection with distortion-aware pose estimation."""
from functools import lru_cache
import cv2
import numpy as np
from .geometry import transform, difference

from .board_config import DEFAULT_BOARD, normalize_board


@lru_cache(maxsize=32)
def _grid_board(nx, ny, length, separation, dictionary_id):
    # Cache immutable geometry to avoid repeated constructor warnings per frame.
    return cv2.aruco.GridBoard((nx, ny), length, separation, cv2.aruco.getPredefinedDictionary(dictionary_id))


def make_board(spec, legacy=None):
    spec = normalize_board(spec)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec['dictionary']))
    if spec['board_type'] == 'gridboard':
        return _grid_board(spec['markers_x'], spec['markers_y'], spec['marker_length_m'],
                           spec['marker_separation_m'], getattr(cv2.aruco, spec['dictionary']))
    board = cv2.aruco.CharucoBoard((spec['squares_x'], spec['squares_y']), spec['square_length_m'], spec['marker_length_m'], dictionary)
    board.setLegacyPattern(bool(legacy))
    return board


def normalize_pixels(pixels, intrinsics):
    import pyrealsense2 as rs
    k = rs.intrinsics()
    k.width, k.height = intrinsics['width'], intrinsics['height']
    matrix = intrinsics['matrix']
    k.fx, k.fy, k.ppx, k.ppy = matrix[0][0], matrix[1][1], matrix[0][2], matrix[1][2]
    model = intrinsics['distortion_model'].removeprefix('distortion.')
    models = {'none', 'brown_conrady', 'inverse_brown_conrady'}
    if model not in models:
        raise ValueError(f"Unsupported distortion model: {model}; do not reinterpret its coefficients")
    k.model = getattr(rs.distortion, model)
    k.coeffs = intrinsics['distortion_coefficients']
    rays = np.asarray([rs.rs2_deproject_pixel_to_point(k, [float(x), float(y)], 1.) for x, y in np.asarray(pixels).reshape(-1, 2)])
    if not np.isfinite(rays).all():
        raise ValueError("Invalid normalized camera rays")
    return rays[:, :2] / rays[:, 2:3]


def residual_pixels(points, normalized, t, intrinsics):
    camera = (t[:3, :3] @ points.T).T + t[:3, 3]
    if np.any(camera[:, 2] <= 0):
        raise ValueError("Board lies behind the camera")
    projected = camera[:, :2] / camera[:, 2:3]
    k = np.asarray(intrinsics['matrix'])
    # Pixel-equivalent residuals in the undistorted image plane.
    return (projected-normalized) * np.array([k[0, 0], k[1, 1]])


def estimate_pose(points, pixels, intrinsics, max_rms_px=1.5):
    obj = np.asarray(points, np.float64).reshape(-1, 3)
    if len(obj) < 6 or np.linalg.matrix_rank(obj-obj.mean(axis=0), tol=1e-7) < 2:
        raise ValueError("Need at least six non-collinear board corners")
    norm = normalize_pixels(pixels, intrinsics)
    count, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, norm, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
    candidates = []
    for rv, tv in zip(rvecs, tvecs):
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = cv2.Rodrigues(rv)[0], tv.ravel()
        try:
            rms = float(np.sqrt(np.mean(np.sum(residual_pixels(obj, norm, t, intrinsics)**2, axis=1))))
            candidates.append((rms, t, rv, tv))
        except ValueError:
            continue
    candidates.sort(key=lambda x: x[0])
    if not count or not candidates:
        raise ValueError("No positive-depth board pose")
    if len(candidates) > 1 and candidates[1][0]-candidates[0][0] < .05 and difference(candidates[0][1], candidates[1][1])[1] > np.deg2rad(10):
        raise ValueError("Ambiguous planar board pose; change the viewing angle")
    _, _, rv, tv = candidates[0]
    rv, tv = cv2.solvePnPRefineLM(obj, norm, np.eye(3), None, rv, tv)
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = cv2.Rodrigues(rv)[0], tv.ravel()
    rms = float(np.sqrt(np.mean(np.sum(residual_pixels(obj, norm, t, intrinsics)**2, axis=1))))
    transform(t)
    if not np.isfinite(rms) or rms > max_rms_px:
        raise ValueError(f"Corner RMS {rms:.2f} px exceeds {max_rms_px:.2f} px")
    return t, rms


def marker_detections(image, spec):
    """Decode markers only; this does not validate metric board geometry."""
    spec = normalize_board(spec)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec['dictionary']))
    params = cv2.aruco.DetectorParameters()
    if spec['board_type'] == 'gridboard':
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        raise ValueError('No board markers detected')
    overlay = image.copy()
    cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
    return gray, corners, ids, overlay


def detect(image, spec, intrinsics, max_rms_px=1.5):
    spec = normalize_board(spec)
    gray, corners, ids, overlay = marker_detections(image, spec)
    if len(set(ids.ravel().tolist())) != len(ids):
        raise ValueError('Duplicate marker IDs in the image; expose only one board')
    if spec['board_type'] == 'gridboard':
        return detect_gridboard(spec, intrinsics, corners, ids, overlay, max_rms_px)
    return detect_charuco(spec, intrinsics, gray, corners, ids, overlay, max_rms_px)


def grid_correspondences(spec, corners, ids):
    """Match each standard GridBoard ID to its four ordered outer corners."""
    board = make_board(spec)
    lookup = dict(zip(board.getIds().ravel(), board.getObjPoints()))
    selected = [(int(i), c) for i, c in zip(ids.ravel(), corners) if int(i) in lookup]
    if len({i for i, _ in selected}) != len(selected):
        raise ValueError('Duplicate marker IDs in the image')
    if len(selected) < 3:
        raise ValueError(f'Only {len(selected)} GridBoard markers; need at least three')
    centers = np.asarray([lookup[i].mean(axis=0) for i, _ in selected])
    if np.linalg.matrix_rank(centers-centers.mean(axis=0), tol=1e-7) < 2:
        raise ValueError('GridBoard marker centers are collinear; expose markers in multiple rows and columns')
    points = np.concatenate([lookup[i] for i, _ in selected])
    pixels = np.concatenate([c.reshape(4, 2) for _, c in selected])
    return [i for i, _ in selected], points, pixels


def detect_gridboard(spec, intrinsics, corners, ids, overlay, max_rms_px):
    matched_ids, points, pixels = grid_correspondences(spec, corners, ids)
    rays = normalize_pixels(pixels, intrinsics)
    h, _ = cv2.findHomography(points[:, :2], rays, method=0)
    if h is None:
        raise ValueError('GridBoard marker layout is degenerate')
    predicted = cv2.perspectiveTransform(points[:, :2].reshape(-1, 1, 2), h).reshape(-1, 2)
    k = intrinsics['matrix']
    layout_rms = float(np.sqrt(np.mean(np.sum(((predicted-rays)*[k[0][0], k[1][1]])**2, axis=1))))
    if not np.isfinite(layout_rms) or layout_rms > 2.5:
        raise ValueError('Marker IDs do not match the configured GridBoard layout')
    t, rms = estimate_pose(points, pixels, intrinsics, max_rms_px)
    return {'board_type': 'gridboard', 'marker_ids': matched_ids,
            'corners_px': pixels.tolist(), 'object_points_m': points.tolist(),
            'T_camera_board': t.tolist(), 'corner_rms_px': rms}, overlay


def detect_charuco(spec, intrinsics, gray, corners, ids, overlay, max_rms_px):
    fits = []
    layouts = []
    preferred = spec['legacy_pattern'] if spec['legacy_pattern'] is not None else False
    for legacy in (preferred, not preferred):
        board = make_board(spec, legacy)
        geometry = np.asarray(board.getObjPoints())
        # Odd-row boards can have identical modern and legacy layouts.
        if any(np.array_equal(geometry, previous) for previous in layouts):
            continue
        layouts.append(geometry)
        lookup = dict(zip(board.getIds().ravel(), board.getObjPoints()))
        selected = [(int(i), c) for i, c in zip(ids.ravel(), corners) if int(i) in lookup]
        if len(selected) < 3:
            continue
        objects = np.concatenate([lookup[i][:, :2] for i, _ in selected])
        pixels = np.concatenate([c.reshape(4, 2) for _, c in selected])
        rays = normalize_pixels(pixels, intrinsics)
        h, _ = cv2.findHomography(objects, rays, method=0)
        if h is None:
            continue
        predicted = cv2.perspectiveTransform(objects.reshape(-1, 1, 2), h).reshape(-1, 2)
        k = intrinsics['matrix']
        error = float(np.sqrt(np.mean(np.sum(((predicted-rays)*[k[0][0], k[1][1]])**2, axis=1))))
        fits.append((error, legacy, board, selected))
    fits.sort(key=lambda x: x[0])
    if not fits or fits[0][0] > 2.5:
        raise ValueError("Marker IDs do not match a consistent board layout")
    if len(fits) > 1 and fits[1][0] < fits[0][0]*2 + .5:
        raise ValueError("Board layout is ambiguous; expose more markers")
    _, legacy, board, selected = fits[0]
    if spec['legacy_pattern'] is not None and legacy != spec['legacy_pattern']:
        raise ValueError("Board layout differs from the session; start a new session")
    marker_ids = np.array([i for i, _ in selected], np.int32).reshape(-1, 1)
    marker_corners = [c for _, c in selected]
    count, cc, ci = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)
    if count < 6 or cc is None:
        raise ValueError(f"Only {count} ChArUco corners; need at least six")
    objects = board.getChessboardCorners()[ci.ravel()]
    t, rms = estimate_pose(objects, cc, intrinsics, max_rms_px)
    cv2.aruco.drawDetectedCornersCharuco(overlay, cc, ci)
    return {'board_type': 'charuco', 'legacy_pattern': legacy, 'marker_ids': marker_ids.ravel().tolist(),
            'charuco_ids': ci.ravel().tolist(), 'corners_px': cc.reshape(-1, 2).tolist(),
            'object_points_m': objects.tolist(), 'T_camera_board': t.tolist(),
            'corner_rms_px': rms}, overlay
