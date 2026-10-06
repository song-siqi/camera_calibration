"""Stationary, operator-triggered acquisition without robot motion commands."""
import copy
import itertools
import time
import cv2
import numpy as np
from .board import detect
from .geometry import difference, inverse, pose_matrix, transform
from .storage import next_sample, samples, read_json, write_json, timestamp, local_path


class SettingsChanged(ValueError):
    pass


def same_settings(a, b):
    if a['model'] != b['model']:
        return False
    for key in ('payload_kg', 'center_of_mass_flange_m', 'tcp_offset_m_rad', 'world_offset_m_rad'):
        if not np.allclose(a[key], b[key], rtol=0, atol=1e-7):
            return False
    return True


def check_observation(obs, expected):
    if not same_settings(obs['settings'], expected):
        raise SettingsChanged("Robot settings changed; quit and create a new session")
    if np.any(np.abs(obs['settings']['world_offset_m_rad']) > 1e-6):
        raise ValueError(f"World offset must be zero: {obs['settings']['world_offset_m_rad']}")
    # MODE_CHANGED (5) blocks motion commands, not read-only pose capture.
    # Fresh, unchanged settings and measured stationarity are checked separately.
    if obs['state'] not in (0, 2, 4, 5) or obs['error'] or obs['warning']:
        raise ValueError(f"Robot is not ready for stationary capture: state={obs['state']}, error={obs['error']}, warning={obs['warning']}")
    transform(obs['T_base_tcp'])


def validate_bracket(observations, frame, expected):
    if len(observations) < 8:
        raise ValueError("Insufficient stationary observations")
    for obs in observations:
        check_observation(obs, expected)
    times = [o['received_monotonic_s'] for o in observations]
    if any(not 0 < b-a < .5 for a, b in zip(times, times[1:])):
        raise ValueError("Robot reports are stale or interrupted")
    if not times[3] < frame['received_monotonic_s'] < times[4]:
        raise ValueError("Image is not bracketed by robot observations")
    changes = [difference(a['T_base_tcp'], b['T_base_tcp']) for a, b in itertools.combinations(observations, 2)]
    translation, rotation = np.max(changes, axis=0)
    if translation > .0002 or rotation > .002:
        raise ValueError("Robot moved during capture; stop teleoperation and retry")
    return {'max_translation_m': float(translation), 'max_rotation_rad': float(rotation)}


def is_duplicate(t, previous):
    return any(difference(t, p)[0] < .01 and difference(t, p)[1] < np.deg2rad(5) for p in previous)


def capture_one(robot, camera, session, identity, board_spec, sleep=time.sleep):
    expected = identity['robot_settings']
    observations = []
    def observe():
        obs = robot.snapshot(after=observations[-1]['received_monotonic_s'] if observations else None)
        check_observation(obs, expected)
        observations.append(obs)
    for _ in range(4):
        observe()
        sleep(.1)
    frame = camera.read(fresh=True)
    # Require the first post-frame report to have arrived after frame receipt.
    obs = robot.snapshot(after=frame['received_monotonic_s'])
    observations.append(obs)
    for _ in range(3):
        sleep(.1)
        observe()
    stationary = validate_bracket(observations, frame, expected)
    detection, _ = detect(frame['image'], board_spec, identity['intrinsics'], identity['max_corner_rms_px'])
    tcp = transform(observations[4]['T_base_tcp'])
    flange = tcp @ inverse(pose_matrix(expected['tcp_offset_m_rad']))
    previous = [read_json(p)['T_base_flange'] for p in samples(session)]
    if is_duplicate(flange, previous):
        raise ValueError("Pose duplicates an existing sample; change position or orientation")
    out = next_sample(session)
    if not cv2.imwrite(str(local_path(out / 'color.png')), frame['image']):
        raise RuntimeError("Failed to save image")
    record = {'timestamp_utc': timestamp(), 'image': 'color.png', 'accepted': True,
              'T_base_tcp': tcp.tolist(), 'T_base_flange': flange.tolist(),
              'robot_settings': copy.deepcopy(expected), 'robot_observations': observations,
              'stationary': stationary, 'camera_frame': {k: v for k, v in frame.items() if k != 'image'},
              'hardware_synchronized': False,
              'pairing': 'stationary robot reports bracket a fresh color frame', 'detection': detection}
    write_json(out / 'sample.json', record)
    result_file = local_path(session / 'result.json')
    if result_file.exists():
        result = read_json(result_file)
        result['validation'].update(passed=False, reason='New samples were added; run solve again')
        write_json(result_file, result)
    return out, detection
