import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import cv2
import pytest
from camera_calibration import cli
from camera_calibration.board import DEFAULT_BOARD
from camera_calibration.geometry import pose_matrix
from camera_calibration.storage import ROOT, read_json, write_json
from test_board import INTRINSICS, render_board
from test_gridboard import GRID_SPEC, render_grid
from test_capture_hardware import bracket


def test_headless_capture_does_not_connect(monkeypatch, capsys):
    monkeypatch.delenv('DISPLAY', raising=False)
    monkeypatch.setattr(cli, 'Robot', lambda *_: pytest.fail('Unexpected robot connection'))
    assert cli.main(['capture']) == 2
    assert 'No desktop DISPLAY' in capsys.readouterr().err


def test_help_and_source_language():
    result = subprocess.run([sys.executable, '-m', 'camera_calibration', '--help'], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0 and '{devices,preview,capture,solve}' in result.stdout
    for path in (ROOT / 'src').rglob('*.py'):
        assert path.read_text().isascii(), path


@pytest.mark.parametrize('robot_state', [4, 5])
@pytest.mark.parametrize('kind', ['charuco', 'gridboard'])
def test_mock_preview_capture_and_quit(local_tmp, monkeypatch, robot_state, kind):
    render = render_grid if kind == 'gridboard' else render_board
    observations, frame, settings = bracket()
    for obs in observations: obs['state'] = robot_state
    calls = []
    class FakeRobot:
        def __init__(self, ip):
            calls.append('robot_connect')
            self.snapshots = iter([copy.deepcopy(observations[0]), copy.deepcopy(observations[0]), *observations, copy.deepcopy(observations[-1])])
        def snapshot(self, **kwargs): return next(self.snapshots)
        def close(self): calls.append('robot_disconnect')
    class FakeCamera:
        identity = {'model': 'fake', 'serial': '123'}
        intrinsics = INTRINSICS
        def __init__(self, serial): calls.append('camera_connect')
        def read(self, **kwargs): return {**frame, 'frame_number': 1, 'image': render(pose_matrix([-.06, -.06, .5, .25, -.2, .15]))}
        def close(self): calls.append('camera_disconnect')
    monkeypatch.setenv('DISPLAY', ':test')
    monkeypatch.setattr(cli, 'Robot', FakeRobot)
    monkeypatch.setattr(cli, 'Camera', FakeCamera)
    monkeypatch.setattr(cli, 'devices', lambda: [{'model': 'fake', 'serial': '123'}])
    monkeypatch.setattr(cv2, 'namedWindow', lambda *args: None)
    monkeypatch.setattr(cv2, 'imshow', lambda *args: None)
    monkeypatch.setattr(cv2, 'getWindowProperty', lambda *args: 1)
    keys = iter([32, ord('q')])
    monkeypatch.setattr(cv2, 'waitKey', lambda *args: next(keys))
    monkeypatch.setattr(cv2, 'destroyAllWindows', lambda: None)
    session = local_tmp / 'session'
    args = ['capture', '--robot-ip', 'fake', '--serial', '123', '--mount', 'fixed', '--session', str(session)]
    if kind == 'gridboard':
        write_json(local_tmp / 'board.json', GRID_SPEC)
        args += ['--board', str(local_tmp / 'board.json')]
    assert cli.main(args) == 0
    assert read_json(session / 'sample_001/sample.json')['accepted']
    assert calls == ['robot_connect', 'camera_connect', 'camera_disconnect', 'robot_disconnect']
    if kind == 'gridboard':
        before = (session / 'session.json').read_bytes()
        keys = iter([32, ord('q')])
        monkeypatch.setattr('builtins.input', lambda _: 'yes')
        assert cli.main(['capture', '--session', str(session)]) == 0
        assert (session / 'session.json').read_bytes() == before
        assert len(list(session.glob('sample_*/sample.json'))) == 1
        assert 'duplicates' in (session / 'rejections.jsonl').read_text()


def test_bad_board_config_fails_before_device_access(local_tmp, monkeypatch, capsys):
    monkeypatch.setenv('DISPLAY', ':test')
    monkeypatch.setattr(cli, 'devices', lambda: pytest.fail('Unexpected camera enumeration'))
    monkeypatch.setattr(cli, 'Robot', lambda *_: pytest.fail('Unexpected robot connection'))
    write_json(local_tmp / 'board.json', {**GRID_SPEC, 'square_length_m': .04})
    assert cli.main(['capture', '--board', str(local_tmp / 'board.json')]) == 2
    assert 'Unsupported fields for gridboard' in capsys.readouterr().err
