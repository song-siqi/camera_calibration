"""English CLI and local OpenCV preview."""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from .board import DEFAULT_BOARD, detect
from .board_config import normalize_board
from .capture import capture_one, check_observation, SettingsChanged
from .hardware import Camera, Robot, devices
from .solver import solve
from .storage import ROOT, local_path, open_session, read_json, samples, record_rejection


def positive(value):
    v = float(value)
    if not np.isfinite(v) or v <= 0:
        raise argparse.ArgumentTypeError('Expected a positive finite number')
    return v


def parser():
    p = argparse.ArgumentParser(description='Read-only xArm6 / RealSense camera calibration')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('devices', help='List connected RealSense cameras')
    v = sub.add_parser('preview', help='View a camera without connecting to the robot or saving samples')
    v.add_argument('--serial', help='Camera serial; select interactively if omitted')
    c = sub.add_parser('capture', help='Open preview and capture stationary samples with SPACE')
    c.add_argument('--robot-ip')
    c.add_argument('--serial')
    c.add_argument('--name', help='Camera name, such as fixed_main or wrist')
    c.add_argument('--mount', choices=['fixed', 'wrist'])
    c.add_argument('--session', type=Path, help='New or existing session directory inside this repository')
    c.add_argument('--board', type=Path, help='ChArUco or GridBoard JSON with board_type; defaults to 4x4 ChArUco, 30 mm squares, 22 mm markers')
    c.add_argument('--max-corner-rms-px', type=positive)
    s = sub.add_parser('solve', help='Solve a saved session offline and write result.json')
    s.add_argument('--session', type=Path, required=True)
    s.add_argument('--max-corner-rms-px', type=positive, default=1.5)
    s.add_argument('--max-translation-mm', type=positive, default=5.)
    s.add_argument('--max-rotation-deg', type=positive, default=2.)
    return p


def select_device(serial):
    found = devices()
    if not found:
        raise RuntimeError('No RealSense cameras found; check USB and permissions')
    for i, d in enumerate(found, 1):
        print(f"{i}. {d['model']}  serial={d['serial']}")
    if serial:
        if serial not in [d['serial'] for d in found]:
            raise ValueError(f'Selected camera {serial} is not connected')
        return serial
    try:
        index = int(input('Select camera number: ')) - 1
        if index < 0:
            raise ValueError()
        return found[index]['serial']
    except (ValueError, IndexError):
        raise ValueError('Invalid camera selection') from None


def preview(args):
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        raise RuntimeError('No desktop DISPLAY. Run preview in a terminal on the robot computer desktop.')
    serial = select_device(args.serial)
    camera = None
    try:
        camera = Camera(serial)
        title = f"Camera Preview - {serial} - Q: quit"
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 960, 720)
        print(f"Previewing {camera.identity['model']} ({serial}). Q / Esc: quit. No robot connection; no files saved.", flush=True)
        first = True
        while True:
            image = camera.read()['image']
            cv2.putText(image, f"{serial} | Q: quit", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(title, image)
            key = cv2.waitKey(1) & 255
            if first:
                print('Live preview window is open.', flush=True)
                first = False
            if key in (ord('q'), ord('Q'), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        try:
            if camera:
                camera.close()
        finally:
            cv2.destroyAllWindows()


def capture(args):
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        raise RuntimeError('No desktop DISPLAY. Run capture in a terminal on the robot computer desktop; devices and solve work without a display.')
    session = local_path(args.session or ROOT / 'sessions' / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    old = read_json(session / 'session.json')['identity'] if (session / 'session.json').exists() else {}
    board = normalize_board(read_json(args.board) if args.board else old.get('board', DEFAULT_BOARD))
    print('Board configuration (metres):')
    print(json.dumps(board, indent=2))
    ip = args.robot_ip or old.get('robot_ip') or input('Robot IP: ').strip()
    serial = select_device(args.serial or old.get('camera', {}).get('serial'))
    mount = args.mount or old.get('mount_type') or input('Mount type (fixed/wrist): ').strip()
    if mount not in ('fixed', 'wrist'):
        raise ValueError('Mount type must be fixed or wrist')
    name = args.name or old.get('camera', {}).get('name') or mount
    threshold = args.max_corner_rms_px or old.get('max_corner_rms_px', 1.5)
    print('FIXED: attach the board rigidly to the end effector.' if mount == 'fixed' else 'WRIST: fix the board on the table; keep it stationary.')
    print('Move the robot with your external teleoperation tool. Stop sending motion commands before pressing SPACE.')
    if old and input('Resume only if the camera and board mounts are unchanged. Type yes to confirm: ').strip().lower() != 'yes':
        raise ValueError('Resume cancelled; create a new session after changing mounts')
    robot = camera = None
    try:
        robot = Robot(ip)
        initial = robot.snapshot()
        check_observation(initial, initial['settings'])
        if initial['state'] == 5:
            print('Robot state 5 (MODE_CHANGED): read-only capture is allowed after stationarity checks. Robot state will not be changed.')
        print('Robot settings (metres / radians / kg):')
        print(json.dumps(initial['settings'], indent=2))
        camera = Camera(serial)
        identity = {'camera': {**camera.identity, 'name': name}, 'intrinsics': camera.intrinsics,
                    'mount_type': mount, 'robot_ip': ip, 'robot_settings': initial['settings'],
                    'board': board, 'max_corner_rms_px': threshold}
        session = open_session(session, identity)
        active_board = copy.deepcopy(board)
        previous = samples(session)
        if previous and board['board_type'] == 'charuco':
            active_board['legacy_pattern'] = read_json(previous[0])['detection']['legacy_pattern']
        title = 'Camera Calibration - SPACE: capture | Q: quit'
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        print(f'Session: {session}\nSPACE: capture   Q: quit\nTarget: 15 distinct poses with rotations about different axes.')
        message = 'Ready. Move, stop, then press SPACE.'
        count = len(previous)
        last_robot_check = 0.
        paused = False
        while True:
            image = np.zeros((480, 640, 3), np.uint8)
            ready = False
            try:
                frame = camera.read()
                image = frame['image']
                try:
                    detection, image = detect(image, active_board, identity['intrinsics'], threshold)
                    ready = True
                    quality = f"Markers: {len(detection['marker_ids'])}  Corners: {len(detection['corners_px'])}  RMS: {detection['corner_rms_px']:.2f}px"
                except (ValueError, cv2.error) as exc:
                    quality = str(exc)
                if time.monotonic()-last_robot_check > .2:
                    obs = robot.snapshot()
                    check_observation(obs, identity['robot_settings'])
                    last_robot_check = time.monotonic()
            except SettingsChanged as exc:
                paused = True
                message = str(exc)
                quality = 'Session paused. Quit and start a new session.'
            except (ValueError, RuntimeError) as exc:
                quality = str(exc)
                ready = False
            lines = [f"{name} | {mount} | {board['board_type']} | Samples: {count}/15", quality, message, 'SPACE: capture | Q: quit']
            for i, line in enumerate(lines):
                cv2.putText(image, line[:88], (8, 22+i*23), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(title, image)
            key = cv2.waitKey(1) & 255
            if key in (ord('q'), ord('Q'), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 32:
                if paused or not ready:
                    message = 'Rejected: ' + ('session settings changed' if paused else quality)
                    print(message)
                    record_rejection(session, message)
                    continue
                try:
                    out, detected = capture_one(robot, camera, session, identity, active_board)
                    if board['board_type'] == 'charuco':
                        active_board['legacy_pattern'] = detected['legacy_pattern']
                    count += 1
                    message = f"Saved {out.name}: {len(detected['corners_px'])} corners, RMS {detected['corner_rms_px']:.2f}px"
                    print(message)
                except SettingsChanged as exc:
                    paused = True
                    message = str(exc)
                    print(message)
                    record_rejection(session, message)
                except (ValueError, RuntimeError, cv2.error) as exc:
                    message = f'Rejected: {exc}'
                    print(message)
                    record_rejection(session, message)
        print(f'Saved {count} samples. Solve with:\ncamera-calibration solve --session {session}')
    finally:
        try:
            if camera:
                camera.close()
        finally:
            if robot:
                robot.close()
            cv2.destroyAllWindows()


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'devices':
            print(json.dumps(devices(), indent=2))
        elif args.command == 'preview':
            preview(args)
        elif args.command == 'capture':
            capture(args)
        else:
            result = solve(args.session, args.max_corner_rms_px, args.max_translation_mm/1000, args.max_rotation_deg)
            print(json.dumps({'result': str(local_path(args.session) / 'result.json'), 'validation': result['validation']}, indent=2))
            return 0 if result['validation']['passed'] else 2
        return 0
    except KeyboardInterrupt:
        print('Stopped by operator.')
        return 130
    except (ValueError, RuntimeError, OSError, KeyError, cv2.error) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
