import cv2
import numpy as np
import pytest
from camera_calibration.geometry import pose_matrix, inverse, sdk_pose, difference, check_rotation_coverage
from camera_calibration.solver import fit, validate


def synthetic(mount, count=18):
    rng = np.random.default_rng(42)
    x = pose_matrix([.12, -.08, .2, .15, -.2, .1])
    y = pose_matrix([.01, .02, .12, -.1, .05, .2])
    aa = [pose_matrix([*rng.uniform(-.25, .25, 3), *rng.uniform(-.8, .8, 3)]) for _ in range(count)]
    bb = [inverse(x) @ (a if mount == 'fixed' else inverse(a)) @ y for a in aa]
    return x, y, aa, bb


@pytest.mark.parametrize('mount', ['fixed', 'wrist'])
def test_known_answer_and_leave_one_out(mount):
    x, y, aa, bb = synthetic(mount)
    solved, board = fit(aa, bb, mount)
    assert np.allclose(solved, x, atol=1e-9)
    assert np.allclose(board, y, atol=1e-9)
    report = validate(aa, bb, mount)
    assert report['translation_rms_m'] < 1e-9
    assert report['rotation_rms_rad'] < 1e-7


def test_bad_sample_fails_validation():
    _, _, aa, bb = synthetic('fixed')
    bb[3] = bb[3].copy()
    bb[3][0, 3] += .03
    assert validate(aa, bb, 'fixed')['translation_rms_m'] > .005


def test_nonzero_tcp_rotation_and_mm_conversion():
    flange = pose_matrix([.3, -.2, .4, .2, .4, -.3])
    offset = pose_matrix([.01, -.02, .172, .1, -.2, .3])
    tcp = flange @ offset
    assert np.allclose(tcp @ inverse(offset), flange)
    x = pose_matrix([.08, 0, .02, .1, .2, 0])
    assert np.allclose(tcp @ (inverse(offset) @ x), flange @ x)
    assert np.allclose(sdk_pose([0, 0, 172, .1, -.2, .3]), pose_matrix([0, 0, .172, .1, -.2, .3]))


def test_parallel_rotations_rejected():
    with pytest.raises(ValueError, match='parallel'):
        check_rotation_coverage([pose_matrix([0, 0, 0, 0, 0, angle]) for angle in np.linspace(-1, 1, 15)])
