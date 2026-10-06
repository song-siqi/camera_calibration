import copy
import pytest
from camera_calibration.board_config import DEFAULT_BOARD, normalize_board
from camera_calibration.storage import open_session, read_json, write_json
from test_gridboard import GRID_SPEC


@pytest.mark.parametrize('kind,key,value', [
    ('gridboard', 'markers_x', 0), ('gridboard', 'markers_y', 3.0),
    ('gridboard', 'markers_x', True), ('gridboard', 'marker_separation_m', 0),
    ('gridboard', 'marker_length_m', float('nan')),
    ('gridboard', 'marker_separation_m', float('inf')),
    ('gridboard', 'square_length_m', .04), ('gridboard', 'legacy_pattern', None),
    ('charuco', 'markers_x', 3), ('charuco', 'legacy_pattern', 1),
    ('charuco', 'marker_length_m', .1), ('charuco', 'dictionary', 'DICT_UNKNOWN'),
    ('gridboard', 'dictionary', 'DICT_UNKNOWN'), ('gridboard', 'board_type', 'auto'),
    ('gridboard', 'markers_x', 100),
])
def test_invalid_board_configuration(kind, key, value):
    spec = dict(GRID_SPEC if kind == 'gridboard' else DEFAULT_BOARD)
    spec[key] = value
    with pytest.raises(ValueError):
        normalize_board(spec)


def test_missing_fields_and_legacy_defaults():
    spec = dict(GRID_SPEC)
    del spec['marker_separation_m']
    with pytest.raises(ValueError, match='marker_separation_m'):
        normalize_board(spec)
    old = dict(DEFAULT_BOARD)
    del old['board_type']
    del old['legacy_pattern']
    assert normalize_board(old) == DEFAULT_BOARD
    assert 'board_type' not in old


def test_old_session_resume_without_rewriting(local_tmp):
    old = dict(DEFAULT_BOARD)
    del old['board_type']
    manifest = {'created_utc': 'old-session', 'identity': {'board': old, 'serial': 'test'}}
    write_json(local_tmp / 'session.json', manifest)
    before = (local_tmp / 'session.json').read_bytes()
    open_session(local_tmp, {'board': DEFAULT_BOARD, 'serial': 'test'})
    assert (local_tmp / 'session.json').read_bytes() == before
    with pytest.raises(ValueError, match='changed'):
        open_session(local_tmp, {'board': GRID_SPEC, 'serial': 'test'})
    with pytest.raises(ValueError, match='changed'):
        open_session(local_tmp, {'board': {**DEFAULT_BOARD, 'square_length_m': .05}, 'serial': 'test'})
    assert (local_tmp / 'session.json').read_bytes() == before


def test_new_session_normalizes_legacy_board(local_tmp):
    old = dict(DEFAULT_BOARD)
    del old['board_type']
    open_session(local_tmp, {'board': old})
    assert read_json(local_tmp / 'session.json')['identity']['board']['board_type'] == 'charuco'
