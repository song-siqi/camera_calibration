import copy
import struct
import time
from types import SimpleNamespace
import numpy as np
import pytest
from camera_calibration.capture import validate_bracket, is_duplicate, SettingsChanged
from camera_calibration.geometry import pose_matrix
from camera_calibration.hardware import parse_report, Robot, next_unique
from camera_calibration.storage import open_session, write_json, read_json, local_path, next_sample


def packet():
    data = bytearray(417)
    data[4] = 4
    struct.pack_into('<6f', data, 35, 300, 20, 200, .1, .2, .3)
    struct.pack_into('<6f', data, 91, 0, 0, 172, 0, 0, 0)
    struct.pack_into('<4f', data, 115, 1.263, -13.22, -2.74, 19.25)
    return data


def bracket():
    observation = parse_report(packet())
    observations = [copy.deepcopy(observation) for _ in range(8)]
    for i, o in enumerate(observations):
        o['received_monotonic_s'] = i*.1
    return observations, {'received_monotonic_s': .35}, observation['settings']


def test_real_report_units_and_complete_packet():
    state = parse_report(packet())
    assert state['settings']['payload_kg'] == pytest.approx(1.263)
    assert state['settings']['center_of_mass_flange_m'][0] == pytest.approx(-.01322)
    assert state['settings']['tcp_offset_m_rad'][2] == pytest.approx(.172)
    assert np.array(state['T_base_tcp'])[0, 3] == pytest.approx(.3)
    with pytest.raises(ValueError, match='complete'):
        parse_report(packet()[:150])


@pytest.mark.parametrize('robot_state', [4, 5])
def test_stationary_and_duplicate(robot_state):
    obs, frame, settings = bracket()
    for o in obs: o['state'] = robot_state
    assert validate_bracket(obs, frame, settings)['max_translation_m'] == 0
    t = pose_matrix([.3, .1, .2, .1, .2, .3])
    assert is_duplicate(t, [t])
    assert not is_duplicate(pose_matrix([.32, .1, .2, .1, .2, .3]), [t])


@pytest.mark.parametrize('change', ['motion', 'rotation', 'state', 'error', 'warning', 'stale', 'frame', 'settings', 'world'])
@pytest.mark.parametrize('robot_state', [4, 5])
def test_bad_bracket_rejected(change, robot_state):
    obs, frame, settings = bracket()
    for o in obs: o['state'] = robot_state
    settings = copy.deepcopy(settings)
    if change == 'motion': obs[-1]['T_base_tcp'][0][3] += .001
    if change == 'rotation': obs[-1]['T_base_tcp'] = pose_matrix([.3, .02, .2, .2, .2, .3]).tolist()
    if change == 'state': obs[-1]['state'] = 1
    if change == 'error': obs[-1]['error'] = 31
    if change == 'warning': obs[-1]['warning'] = 1
    if change == 'stale': obs[-1]['received_monotonic_s'] = .6
    if change == 'frame': frame['received_monotonic_s'] = -.1
    if change == 'settings': obs[-1]['settings']['payload_kg'] += .1
    if change == 'world':
        settings['world_offset_m_rad'][0] = .01
        for o in obs: o['settings'] = copy.deepcopy(settings)
    with pytest.raises(ValueError): validate_bracket(obs, frame, settings)


def test_frame_retry_and_timeout():
    class Frame:
        def __init__(self, n): self.n = n
        def get_frame_number(self): return self.n
    frames = iter([Frame(2), Frame(2), Frame(3)])
    assert next_unique(lambda: next(frames), 2).n == 3
    with pytest.raises(RuntimeError, match='No new camera'):
        next_unique(lambda: Frame(2), 2, timeout_s=.001)


def test_session_resume_mismatch_and_output_scope(local_tmp):
    identity = {'serial': 'camera-a'}
    assert open_session(local_tmp, identity) == local_tmp
    assert open_session(local_tmp, identity) == local_tmp
    first = next_sample(local_tmp)
    assert first.name == 'sample_001'
    assert next_sample(local_tmp).name == 'sample_002'
    with pytest.raises(ValueError, match='changed'):
        open_session(local_tmp, {'serial': 'camera-b'})
    with pytest.raises(ValueError, match='inside'):
        local_path('/tmp/outside.json')
    (local_tmp / 'escape').symlink_to('/tmp', target_is_directory=True)
    with pytest.raises(ValueError, match='inside'):
        write_json(local_tmp / 'escape' / 'test.json', {})


def test_sdk_adapter_read_only_and_disconnect():
    calls = []
    class FakeAPI:
        def __init__(self, *args, **kwargs):
            assert kwargs['do_not_open'] and not kwargs['keep_heart']
            self._arm = SimpleNamespace(_is_old_protocol=False, clean_warn=lambda: calls.append('clean_warn'), _handle_report_data=lambda data: None)
            self.connected, self.axis = False, 6
        def connect(self):
            calls.append('connect')
            self.connected = True
            self._arm.clean_warn()
            self._arm._handle_report_data(packet())
        def disconnect(self):
            calls.append('disconnect')
            self.connected = False
    robot = Robot('test', FakeAPI)
    assert robot.snapshot()['settings']['payload_kg'] > 1
    robot.close()
    with pytest.raises(RuntimeError, match='disconnected'): robot.snapshot()
    assert calls == ['connect', 'disconnect']
