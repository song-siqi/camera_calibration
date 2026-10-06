import copy
import json
import cv2
import numpy as np
import pytest
from camera_calibration.board import DEFAULT_BOARD
from camera_calibration.capture import capture_one
from camera_calibration.geometry import pose_matrix, inverse, difference
from camera_calibration.solver import solve
from camera_calibration.storage import open_session, next_sample, write_json, read_json
from test_board import INTRINSICS, render_board
from test_gridboard import GRID_SPEC, render_grid
from test_capture_hardware import bracket


def identity(mount='fixed'):
    _, _, settings = bracket()
    settings['tcp_offset_m_rad'] = [.01, -.02, .172, .1, -.2, .3]
    return {'camera': {'model': 'synthetic', 'serial': 'synthetic', 'name': 'test'},
            'intrinsics': INTRINSICS, 'mount_type': mount, 'robot_ip': 'synthetic',
            'robot_settings': settings, 'board': DEFAULT_BOARD.copy(), 'max_corner_rms_px': 1.5}


@pytest.mark.parametrize('mount', ['fixed', 'wrist'])
@pytest.mark.parametrize('kind', ['charuco', 'gridboard', 'legacy_charuco'])
def test_images_to_json_known_extrinsics(local_tmp, mount, kind):
    config = identity(mount)
    if kind == 'gridboard':
        config['board'] = GRID_SPEC.copy()
    session = open_session(local_tmp, config)
    if kind == 'legacy_charuco':
        manifest = read_json(session / 'session.json')
        del manifest['identity']['board']['board_type']
        write_json(session / 'session.json', manifest)
    manifest_before = (session / 'session.json').read_bytes()
    render = render_grid if kind == 'gridboard' else render_board
    x = pose_matrix([.1, -.05, .2, .1, -.15, .2])
    y = pose_matrix([0, 0, .12, -.1, .2, .1])
    offset = pose_matrix(config['robot_settings']['tcp_offset_m_rad'])
    rng = np.random.default_rng(81)
    for _ in range(18):
        b = pose_matrix([rng.uniform(-.09, -.04), rng.uniform(-.09, -.04), rng.uniform(.4, .58),
                         rng.uniform(-.5, .5), rng.uniform(-.5, .5), rng.uniform(-.3, .3)])
        a = x @ b @ inverse(y) if mount == 'fixed' else y @ inverse(b) @ inverse(x)
        out = next_sample(session)
        cv2.imwrite(str(out / 'color.png'), render(b))
        observations, frame, _ = bracket()
        for obs in observations:
            obs['settings'] = copy.deepcopy(config['robot_settings'])
            obs['T_base_tcp'] = (a @ offset).tolist()
        write_json(out / 'sample.json', {'accepted': True, 'image': 'color.png',
                   'robot_settings': config['robot_settings'], 'robot_observations': observations,
                   'camera_frame': frame, 'T_base_tcp': (a @ offset).tolist(), 'T_base_flange': a.tolist()})
    result = solve(session)
    assert result['validation']['passed'], result['validation']
    assert result['calibration']['sample_count'] >= 12
    expected = x if mount == 'fixed' else inverse(offset) @ x
    t, r = difference(expected, result['extrinsics']['matrix'])
    assert t < .004 and r < np.deg2rad(1)
    if mount == 'wrist':
        assert difference(x, result['extrinsics']['flange']['matrix'])[0] < .004
    assert read_json(session / 'result.json') == result
    assert result['calibration']['board']['board_type'] == ('gridboard' if kind == 'gridboard' else 'charuco')
    assert (session / 'session.json').read_bytes() == manifest_before
    assert result['extrinsics']['target_frame'] == ('robot_base' if mount == 'fixed' else 'robot_tcp')


def test_insufficient_samples_exports_failure(local_tmp):
    session = open_session(local_tmp, identity())
    result = solve(session)
    assert result['extrinsics'] is None
    assert not result['validation']['passed']
    assert 'at least 12' in result['validation']['reason']


@pytest.mark.parametrize('kind', ['charuco', 'gridboard'])
def test_operator_capture_resume_duplicate_and_result_invalidation(local_tmp, kind):
    config = identity()
    if kind == 'gridboard':
        config['board'] = GRID_SPEC.copy()
    render = render_grid if kind == 'gridboard' else render_board
    session = open_session(local_tmp, config)
    observations, frame, _ = bracket()
    for o in observations:
        o['settings'] = copy.deepcopy(config['robot_settings'])
    class FakeRobot:
        def __init__(self): self.values = iter(copy.deepcopy(observations))
        def snapshot(self, **kwargs): return next(self.values)
    class FakeCamera:
        def read(self, fresh=False):
            assert fresh
            return {**frame, 'image': render(pose_matrix([-.06, -.06, .5, .25, -.2, .15])), 'frame_number': 10}
    write_json(session / 'result.json', {'validation': {'passed': True}})
    out, detection = capture_one(FakeRobot(), FakeCamera(), session, config, config['board'], sleep=lambda _: None)
    assert (out / 'color.png').exists()
    assert detection['board_type'] == kind
    assert read_json(out / 'sample.json')['hardware_synchronized'] is False
    assert not read_json(session / 'result.json')['validation']['passed']
    assert open_session(session, config) == session
    with pytest.raises(ValueError, match='duplicates'):
        capture_one(FakeRobot(), FakeCamera(), session, config, config['board'], sleep=lambda _: None)
