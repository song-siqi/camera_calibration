"""Explicit board geometry with backwards-compatible ChArUco defaults."""
import math
import cv2

DEFAULT_BOARD = {'board_type': 'charuco', 'squares_x': 4, 'squares_y': 4,
                 'square_length_m': .03, 'marker_length_m': .022,
                 'dictionary': 'DICT_4X4_50', 'legacy_pattern': None}


def normalize_board(value):
    if not isinstance(value, dict):
        raise ValueError('Board configuration must be a JSON object')
    spec = dict(value)
    kind = spec.setdefault('board_type', 'charuco')
    if kind not in ('charuco', 'gridboard'):
        raise ValueError('board_type must be charuco or gridboard')
    common = {'board_type', 'dictionary', 'marker_length_m'}
    specific = ({'squares_x', 'squares_y', 'square_length_m', 'legacy_pattern'}
                if kind == 'charuco' else {'markers_x', 'markers_y', 'marker_separation_m'})
    unknown = set(spec) - common - specific
    if unknown:
        raise ValueError(f"Unsupported fields for {kind}: {', '.join(sorted(unknown))}")
    required = (common | specific) - {'legacy_pattern'}
    missing = required - set(spec)
    if missing:
        raise ValueError(f"Missing fields for {kind}: {', '.join(sorted(missing))}")
    dimensions = ('squares_x', 'squares_y') if kind == 'charuco' else ('markers_x', 'markers_y')
    for key in dimensions:
        if type(spec[key]) is not int or spec[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    lengths = ('marker_length_m', 'square_length_m' if kind == 'charuco' else 'marker_separation_m')
    for key in lengths:
        value = spec[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be a positive finite length in metres')
        spec[key] = float(value)
    name = spec['dictionary']
    if not isinstance(name, str) or not name.startswith('DICT_') or not hasattr(cv2.aruco, name):
        raise ValueError(f'Unknown ArUco dictionary: {name}')
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    nx, ny = (spec[key] for key in dimensions)
    if kind == 'charuco':
        if min(nx, ny) < 2:
            raise ValueError('ChArUco requires at least two squares in each direction')
        if spec['marker_length_m'] >= spec['square_length_m']:
            raise ValueError('marker_length_m must be smaller than square_length_m')
        legacy = spec.setdefault('legacy_pattern', None)
        if legacy is not None and type(legacy) is not bool:
            raise ValueError('legacy_pattern must be true, false or null')
        count = nx * ny // 2
    else:
        if min(nx, ny) < 2:
            raise ValueError('GridBoard capture requires at least two rows and two columns')
        count = nx * ny
    if count > len(dictionary.bytesList):
        raise ValueError(f'Board requires {count} marker IDs but {name} has only {len(dictionary.bytesList)}')
    return spec
