import cv2
import numpy as np
import pytest
from camera_calibration.board import DEFAULT_BOARD, make_board, detect, normalize_pixels, estimate_pose
from camera_calibration.geometry import pose_matrix, difference

INTRINSICS = {'width': 640, 'height': 480, 'fps': 30, 'stream': 'color',
              'matrix': [[600., 0, 320.], [0, 600., 240.], [0, 0, 1]],
              'distortion_model': 'distortion.none', 'distortion_coefficients': [0.]*5}


def render_board(t, legacy=False):
    board = make_board(DEFAULT_BOARD, legacy)
    pattern = board.generateImage((600, 600), marginSize=0)
    object_corners = np.array([[0, 0, 0], [.12, 0, 0], [.12, .12, 0], [0, .12, 0]], np.float64)
    corners, _ = cv2.projectPoints(object_corners, cv2.Rodrigues(t[:3, :3])[0], t[:3, 3], np.array(INTRINSICS['matrix']), None)
    h = cv2.getPerspectiveTransform(np.float32([[-.5, -.5], [599.5, -.5], [599.5, 599.5], [-.5, 599.5]]), (corners.reshape(4, 2)*4+1.5).astype(np.float32))
    large = cv2.warpPerspective(pattern, h, (2560, 1920), borderValue=255)
    image = cv2.resize(large, (640, 480), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize('legacy', [False, True])
def test_layout_detection_and_pnp(legacy):
    t = pose_matrix([-.06, -.06, .5, .25, -.2, .15])
    result, _ = detect(render_board(t, legacy), DEFAULT_BOARD, INTRINSICS)
    assert result['legacy_pattern'] == legacy
    assert len(result['charuco_ids']) == 9
    assert difference(result['T_camera_board'], t)[0] < .005
    assert difference(result['T_camera_board'], t)[1] < np.deg2rad(2)


def test_wrong_layout_and_blank_rejected():
    t = pose_matrix([-.06, -.06, .5, .25, -.2, .15])
    with pytest.raises(ValueError, match='layout differs'):
        detect(render_board(t, True), {**DEFAULT_BOARD, 'legacy_pattern': False}, INTRINSICS)
    with pytest.raises(ValueError, match='No board'):
        detect(np.zeros((480, 640, 3), np.uint8), DEFAULT_BOARD, INTRINSICS)


def test_inverse_brown_normalization_uses_realsense():
    import pyrealsense2 as rs
    spec = {**INTRINSICS, 'distortion_model': 'distortion.inverse_brown_conrady', 'distortion_coefficients': [-.05, .06, -.001, .0005, -.02]}
    pixels = np.array([[50., 60.], [510., 370.], [320., 240.]])
    k = rs.intrinsics()
    k.width, k.height, k.fx, k.fy, k.ppx, k.ppy = 640, 480, 600., 600., 320., 240.
    k.model, k.coeffs = rs.distortion.inverse_brown_conrady, spec['distortion_coefficients']
    expected = np.array([rs.rs2_deproject_pixel_to_point(k, p.tolist(), 1.)[:2] for p in pixels])
    assert np.allclose(normalize_pixels(pixels, spec), expected)
    assert not np.allclose(normalize_pixels(pixels, spec), normalize_pixels(pixels, INTRINSICS))


def test_unknown_distortion_and_collinear_rejected():
    with pytest.raises(ValueError, match='Unsupported distortion'):
        normalize_pixels([[1, 2]], {**INTRINSICS, 'distortion_model': 'unknown'})
    points = np.array([[i, 0, 0] for i in range(6)], float)
    with pytest.raises(ValueError, match='non-collinear'):
        estimate_pose(points, np.zeros((6, 2)), INTRINSICS)


@pytest.mark.parametrize('legacy', [None, False, True])
def test_odd_row_board_equivalent_layouts(legacy):
    spec = {**DEFAULT_BOARD, 'squares_x': 6, 'squares_y': 5,
            'square_length_m': .04, 'marker_length_m': .03, 'legacy_pattern': legacy}
    board = make_board(spec, False)
    image = np.full((480, 640, 3), 255, np.uint8)
    image[40:440, 80:560] = cv2.cvtColor(board.generateImage((480, 400)), cv2.COLOR_GRAY2BGR)
    result, _ = detect(image, spec, INTRINSICS)
    assert len(result['marker_ids']) == 15
    assert len(result['charuco_ids']) == 20
    assert result['legacy_pattern'] == bool(legacy)
    with pytest.raises(ValueError, match='layout'):
        detect(image, {**spec, 'squares_x': 5, 'squares_y': 6}, INTRINSICS)
