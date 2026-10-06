"""Park hand-eye solve and leave-one-out validation."""
from pathlib import Path
import cv2
import numpy as np
from .board import detect
from .board_config import normalize_board
from .capture import same_settings, validate_bracket
from .geometry import transform, inverse, pose_matrix, difference, mean_transform, check_rotation_coverage
from .storage import local_path, read_json, write_json, samples, timestamp


def fit(flange_poses, board_poses, mount_type):
    if mount_type not in ('fixed', 'wrist'):
        raise ValueError("Unknown mount type")
    effective = [inverse(a) if mount_type == 'fixed' else transform(a) for a in flange_poses]
    check_rotation_coverage(effective)
    r, t = cv2.calibrateHandEye([a[:3, :3] for a in effective], [a[:3, 3] for a in effective],
                              [b[:3, :3] for b in board_poses], [b[:3, 3] for b in board_poses],
                              method=cv2.CALIB_HAND_EYE_PARK)
    x = np.eye(4)
    x[:3, :3], x[:3, 3] = r, t.ravel()
    transform(x)
    y = mean_transform([a @ x @ b for a, b in zip(effective, board_poses)])
    return x, y


def validate(flange_poses, board_poses, mount_type):
    rows = []
    for i in range(len(flange_poses)):
        aa = flange_poses[:i]+flange_poses[i+1:]
        bb = board_poses[:i]+board_poses[i+1:]
        x, y = fit(aa, bb, mount_type)
        a = inverse(flange_poses[i]) if mount_type == 'fixed' else flange_poses[i]
        predicted = inverse(x) @ inverse(a) @ y
        translation, rotation = difference(predicted, board_poses[i])
        rows.append({'index': i, 'translation_m': translation, 'rotation_rad': rotation})
    return {'translation_rms_m': float(np.sqrt(np.mean([r['translation_m']**2 for r in rows]))),
            'rotation_rms_rad': float(np.sqrt(np.mean([r['rotation_rad']**2 for r in rows]))),
            'leave_one_out': rows}


def solve(session, max_corner_rms_px=1.5, max_translation_m=.005, max_rotation_deg=2.):
    session = local_path(session)
    identity = read_json(session / 'session.json')['identity']
    spec = normalize_board(identity['board'])
    result = {'camera': identity['camera'], 'mount_type': identity['mount_type'],
              'intrinsics': identity['intrinsics'], 'extrinsics': None,
              'robot_settings': identity['robot_settings'],
              'units': {'length': 'm', 'angle': 'rad', 'mass': 'kg'},
              'calibration': {'timestamp_utc': timestamp(), 'board': spec, 'method': 'OpenCV Park',
                              'used_samples': [], 'excluded_samples': [], 'sample_count': 0},
              'validation': {'passed': False, 'thresholds': {'corner_rms_px': max_corner_rms_px,
                              'translation_rms_m': max_translation_m, 'rotation_rms_rad': float(np.deg2rad(max_rotation_deg))},
                             'residual_definition': 'pixel-equivalent error in the undistorted image plane'}}
    aa, bb, rms = [], [], []
    for file in samples(session):
        name = file.parent.name
        try:
            sample = read_json(file)
            if not sample.get('accepted') or not same_settings(sample['robot_settings'], identity['robot_settings']):
                raise ValueError("Sample settings differ from session")
            validate_bracket(sample['robot_observations'], sample['camera_frame'], identity['robot_settings'])
            image_path = local_path(file.parent / sample['image'])
            if image_path.parent != file.parent:
                raise ValueError("Sample image must be inside its sample directory")
            image = cv2.imread(str(image_path))
            if image is None or image.shape[:2] != (identity['intrinsics']['height'], identity['intrinsics']['width']):
                raise ValueError("Missing image or image dimensions differ from intrinsics")
            if 'detection' in sample and sample['detection'].get('board_type', 'charuco') != spec['board_type']:
                raise ValueError('Sample board type differs from session')
            detection, _ = detect(image, spec, identity['intrinsics'], max_corner_rms_px)
            if spec['board_type'] == 'charuco':
                spec['legacy_pattern'] = detection['legacy_pattern']
            a = transform(sample['T_base_tcp']) @ inverse(pose_matrix(identity['robot_settings']['tcp_offset_m_rad']))
            if not np.allclose(a, sample['T_base_flange'], atol=1e-7):
                raise ValueError("Stored flange pose is inconsistent with TCP offset")
            aa.append(a)
            bb.append(transform(detection['T_camera_board']))
            rms.append(detection['corner_rms_px'])
            result['calibration']['used_samples'].append(name)
        except (ValueError, RuntimeError, KeyError, cv2.error) as exc:
            result['calibration']['excluded_samples'].append({'sample': name, 'reason': str(exc)})
    result['calibration']['sample_count'] = len(aa)
    try:
        if len(aa) < 12:
            raise ValueError(f"Only {len(aa)} usable samples; at least 12 required")
        coverage = check_rotation_coverage(aa)
        x, y = fit(aa, bb, identity['mount_type'])
        validation = validate(aa, bb, identity['mount_type'])
        for row, name in zip(validation['leave_one_out'], result['calibration']['used_samples']):
            row['sample'] = name
        result['validation'].update(validation, corner_rms_px=rms, rotation_coverage_singular_values=coverage)
        target = 'robot_base' if identity['mount_type'] == 'fixed' else 'robot_tcp'
        primary = x if identity['mount_type'] == 'fixed' else inverse(pose_matrix(identity['robot_settings']['tcp_offset_m_rad'])) @ x
        result['extrinsics'] = {'source_frame': 'color_optical_frame', 'target_frame': target,
                                'matrix': primary.tolist(), 'convention': 'p_target = T_target_source @ p_source'}
        if identity['mount_type'] == 'wrist':
            result['extrinsics']['flange'] = {'source_frame': 'color_optical_frame', 'target_frame': 'robot_flange', 'matrix': x.tolist()}
        passed = validation['translation_rms_m'] <= max_translation_m and validation['rotation_rms_rad'] <= np.deg2rad(max_rotation_deg)
        result['validation']['passed'] = bool(passed)
        result['validation']['reason'] = 'All numerical checks passed; task accuracy still needs physical verification' if passed else 'Leave-one-out error exceeds limits; inspect per-sample errors and collect better poses'
    except (ValueError, RuntimeError, cv2.error, np.linalg.LinAlgError) as exc:
        result['validation']['reason'] = str(exc)
    write_json(session / 'result.json', result)
    return result
