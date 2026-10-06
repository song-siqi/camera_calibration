import cv2
import numpy as np
import pytest
from camera_calibration.board import make_board, detect, grid_correspondences
from camera_calibration.geometry import pose_matrix, difference
from test_board import INTRINSICS

GRID_SPEC = {'board_type': 'gridboard', 'markers_x': 3, 'markers_y': 3,
             'marker_length_m': .035, 'marker_separation_m': .0075, 'dictionary': 'DICT_4X4_50'}


def render_grid(t, spec=None, hidden=()):
    spec = spec or GRID_SPEC
    board = make_board(spec)
    width, height, _ = board.getRightBottomCorner()
    size = (round(width*5000), round(height*5000))
    pattern = board.generateImage(size, marginSize=0)
    for index in hidden:
        xy = np.asarray(board.getObjPoints()[index])[:, :2]
        lo = np.floor(xy.min(axis=0)*5000-2).astype(int).clip(0)
        hi = np.ceil(xy.max(axis=0)*5000+2).astype(int)
        pattern[lo[1]:hi[1], lo[0]:hi[0]] = 255
    object_corners = np.array([[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0]], float)
    pixels, _ = cv2.projectPoints(object_corners, cv2.Rodrigues(t[:3, :3])[0], t[:3, 3], np.array(INTRINSICS['matrix']), None)
    source = np.float32([[-.5, -.5], [size[0]-.5, -.5], [size[0]-.5, size[1]-.5], [-.5, size[1]-.5]])
    h = cv2.getPerspectiveTransform(source, (pixels.reshape(4, 2)*4+1.5).astype(np.float32))
    large = cv2.warpPerspective(pattern, h, (2560, 1920), borderValue=255)
    return cv2.cvtColor(cv2.resize(large, (640, 480), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize('angle,hidden', [(0., ()), (1.2, ()), (-.7, (0, 2))])
def test_grid_pose_and_occlusion(angle, hidden):
    t = pose_matrix([-.045, -.055, .45, .3, -.25, angle])
    d, _ = detect(render_grid(t, hidden=hidden), GRID_SPEC, INTRINSICS)
    assert d['board_type'] == 'gridboard'
    assert len(d['marker_ids']) == 9-len(hidden)
    assert len(d['corners_px']) == 4*len(d['marker_ids'])
    assert 'charuco_ids' not in d and 'legacy_pattern' not in d
    translation, rotation = difference(d['T_camera_board'], t)
    assert translation < .002 and rotation < np.deg2rad(1)


def test_wrong_grid_layout_and_spacing_rejected():
    t = pose_matrix([-.06, -.06, .4, .25, -.2, .15])
    image = render_grid(t)
    with pytest.raises(ValueError, match='layout'):
        detect(image, {**GRID_SPEC, 'markers_x': 4}, INTRINSICS)
    with pytest.raises(ValueError, match='layout|RMS'):
        detect(image, {**GRID_SPEC, 'marker_separation_m': .02}, INTRINSICS)


@pytest.mark.parametrize('ids,message', [([0, 1], 'at least three'), ([0, 1, 2], 'collinear'), ([0, 0, 3], 'Duplicate')])
def test_insufficient_or_duplicate_grid_markers(ids, message):
    board = make_board(GRID_SPEC)
    corners = [(board.getObjPoints()[i][:, :2]*1000).reshape(1, 4, 2) for i in ids]
    with pytest.raises(ValueError, match=message):
        grid_correspondences(GRID_SPEC, corners, np.array(ids).reshape(-1, 1))


def test_partial_row_is_not_enough_for_grid_capture():
    image = render_grid(pose_matrix([-.06, -.06, .4, .25, -.2, .15]), hidden=range(3, 9))
    with pytest.raises(ValueError, match='collinear'):
        detect(image, GRID_SPEC, INTRINSICS)
