"""Property controls for MATLAB camera-relative local lights.

Provenance
----------
MATLAB R2024b camlight left/headlight properties captured on an asymmetric
surface at three views and data-aspect ratios. The fixture is a retained source
property capture; the second check uses only camera-basis geometric relations.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from chebfunjax._surface_lighting import matlab_camlight_position

_CAPTURE = Path(__file__).parent / 'fixtures' / 'camlight_left_r2024b.json'


def test_r2024b_camera_light_positions_match_native_properties():
    capture = json.loads(_CAPTURE.read_text())
    assert capture['matlab_version'] == '24.2.0.2806996 (R2024b) Update 3'
    assert len(capture['cases']) == 6
    for case in capture['cases']:
        actual = np.asarray(matlab_camlight_position(
            case['camera_position'], case['camera_target'],
            case['data_aspect_ratio'], case['camera_up_vector'],
            case['light_kind']))
        expected = np.asarray(case['light_position'])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-13)
        assert case['light_style'] == 'local'
        if case['light_kind'] == 'headlight':
            np.testing.assert_array_equal(actual, case['camera_position'])


def test_left_light_has_source_camera_relative_direction_under_dar():
    capture = json.loads(_CAPTURE.read_text())
    expected = np.array([-np.sqrt(3)/4, 1/2, 3/4])
    for case in capture['cases']:
        if case['light_kind'] != 'left':
            continue
        dar = np.asarray(case['data_aspect_ratio'])
        camera = np.asarray(case['camera_position'])
        target = np.asarray(case['camera_target'])
        light = np.asarray(case['light_position'])
        eye = (camera-target)/dar
        radius = np.linalg.norm(eye)
        eye /= radius
        up = np.asarray(case['camera_up_vector'])/dar
        up -= np.dot(up, eye)*eye
        up /= np.linalg.norm(up)
        right = np.cross(up, eye)
        right /= np.linalg.norm(right)
        light_ray = (light-target)/dar
        np.testing.assert_allclose(np.linalg.norm(light_ray), radius,
                                   rtol=0, atol=3e-13)
        light_ray /= radius
        np.testing.assert_allclose([np.dot(light_ray, right),
                                    np.dot(light_ray, up),
                                    np.dot(light_ray, eye)],
                                   expected, rtol=0, atol=3e-14)


def test_light_position_errors_are_clear():
    args = ([0, 0, 1], [0, 0, 0], [1, 1, 1], [0, 0, 1])
    with pytest.raises(ValueError, match='position must'):
        matlab_camlight_position(*args, position='camera-ish')
    with pytest.raises(ValueError, match='positive'):
        matlab_camlight_position(args[0], args[1], [1, 0, 1], args[3])
    with pytest.raises(ValueError, match='must differ'):
        matlab_camlight_position([0, 0, 0], args[1], args[2], args[3])
    with pytest.raises(ValueError, match='parallel'):
        matlab_camlight_position([0, 0, 1], [0, 0, 0], [1, 1, 1], [0, 0, 1])
