"""Device adapters. Robot connection never commands motion or changes settings."""
import copy
import struct
import threading
import time
import numpy as np
from .geometry import sdk_pose


def parse_report(data):
    """Decode modern xArm rich report fields; reject incomplete packets."""
    if len(data) < 312:
        raise ValueError("A complete modern rich report is required")
    pose = struct.unpack_from('<6f', data, 35)
    tcp = struct.unpack_from('<6f', data, 91)
    load = struct.unpack_from('<4f', data, 115)
    world = struct.unpack_from('<6f', data, 288)
    if not np.isfinite([*pose, *tcp, *load, *world]).all() or load[0] < 0:
        raise ValueError("Invalid robot report values")
    settings = {'model': 'xArm6', 'payload_kg': load[0],
                'center_of_mass_flange_m': (np.array(load[1:])/1000).tolist(),
                'tcp_offset_m_rad': [v/1000 if i < 3 else v for i, v in enumerate(tcp)],
                'world_offset_m_rad': [v/1000 if i < 3 else v for i, v in enumerate(world)]}
    return {'T_base_tcp': sdk_pose(pose).tolist(), 'settings': settings,
            'state': data[4] & 15, 'error': data[89], 'warning': data[90],
            'received_monotonic_s': time.monotonic()}


class Robot:
    def __init__(self, ip, api_factory=None):
        if api_factory is None:
            from xarm.wrapper import XArmAPI
            api_factory = XArmAPI
        self._lock = threading.Lock()
        self._sample = None
        self._report_error = None
        self.api = api_factory(ip, is_radian=True, do_not_open=True, report_type='rich', keep_heart=False)
        # SDK 1.18.4 otherwise clears warnings during its version handshake.
        self.api._arm.clean_warn = lambda: 0
        native = self.api._arm._handle_report_data

        def on_report(data):
            native(data)
            try:
                if self.api._arm._is_old_protocol:
                    raise ValueError("Legacy robot report protocol is unsupported")
                sample = parse_report(data)
                with self._lock:
                    self._sample, self._report_error = sample, None
            except Exception as exc:
                with self._lock:
                    self._report_error = str(exc)

        self.api._arm._handle_report_data = on_report
        try:
            self.api.connect()
            deadline = time.monotonic() + 5
            while self._sample is None and time.monotonic() < deadline:
                time.sleep(.02)
            self.snapshot()
            if self.api.axis != 6:
                raise ValueError("This tool supports xArm6 only")
        except BaseException:
            self.close()
            raise

    def snapshot(self, after=None, timeout=1.):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.api.connected:
                raise RuntimeError("Robot disconnected")
            with self._lock:
                data, error = copy.deepcopy(self._sample), self._report_error
            if error:
                raise RuntimeError(error)
            if data and time.monotonic()-data['received_monotonic_s'] < .5 and (after is None or data['received_monotonic_s'] > after):
                return data
            time.sleep(.01)
        raise RuntimeError("No fresh complete robot report; check the connection")

    def close(self):
        self.api.disconnect()


def devices():
    import pyrealsense2 as rs
    return [{'model': d.get_info(rs.camera_info.name), 'serial': d.get_info(rs.camera_info.serial_number)}
            for d in rs.context().query_devices()]


def next_unique(wait_frame, last_number, timeout_s=2.):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = wait_frame()
        if frame and frame.get_frame_number() > last_number:
            return frame
    raise RuntimeError("No new camera frame; check USB connection and camera ownership")


class Camera:
    def __init__(self, serial):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipeline = rs.pipeline()
        self.last_number = -1
        self.started = False
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        try:
            profile = self.pipeline.start(cfg)
            self.started = True
            device = profile.get_device()
            self.identity = {'model': device.get_info(rs.camera_info.name), 'serial': serial}
            k = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.intrinsics = {'stream': 'color', 'width': k.width, 'height': k.height, 'fps': 30,
                               'matrix': [[k.fx, 0, k.ppx], [0, k.fy, k.ppy], [0, 0, 1]],
                               'distortion_model': str(k.model), 'distortion_coefficients': list(k.coeffs)}
            for _ in range(30):
                self.read()
        except BaseException as exc:
            self.close()
            raise RuntimeError(f"Cannot start camera {serial}: {exc}. Release it in other applications and check USB.") from exc

    def read(self, fresh=False):
        if fresh:
            # Discard queued exposures and two subsequent frames after the pose bracket starts.
            for _ in range(60):
                if not self.pipeline.poll_for_frames():
                    break
            else:
                raise RuntimeError("Camera buffer did not drain")
            for _ in range(2):
                self._read_frame()
        return self._read_frame()

    def _read_frame(self):
        frame = next_unique(lambda: self.pipeline.wait_for_frames(1000).get_color_frame(), self.last_number)
        self.last_number = frame.get_frame_number()
        return {'image': np.asanyarray(frame.get_data()).copy(), 'frame_number': self.last_number,
                'sensor_timestamp_ms': frame.get_timestamp(), 'received_monotonic_s': time.monotonic()}

    def close(self):
        if self.started:
            self.pipeline.stop()
            self.started = False
