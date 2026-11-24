import json
import re
import socket
import threading
import time

import serial

DEFAULT_PORT = "/dev/cu.usbserial-11210"
DEFAULT_HOST = "elegoo-car.local"
REPLY = re.compile(rb"\{([^{}]*)\}")
HEARTBEAT = b"{Heartbeat}"
DIRECTIONS = {"left": 2, "right": 1, "forward": 3, "back": 4}
MAX_RANGE_CM = 150


class WifiLink:
    def __init__(self, host=DEFAULT_HOST, port=100, connect_timeout=5):
        self.sock = socket.create_connection((host, port), timeout=connect_timeout)
        self.sock.settimeout(0.2)
        self._buf = bytearray()
        self._buf_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._alive = True
        self.error = None
        self._pump = threading.Thread(target=self._run, daemon=True)
        self._pump.start()

    def _run(self):
        while self._alive:
            try:
                chunk = self.sock.recv(1024)
            except socket.timeout:
                continue
            except OSError as e:
                self.error = e
                break
            if not chunk:
                self.error = ConnectionError("car closed the connection")
                break
            with self._buf_lock:
                self._buf += chunk
                beats = self._buf.count(HEARTBEAT)
                if beats:
                    self._buf = self._buf.replace(HEARTBEAT, b"")
            for _ in range(beats):
                self.write(HEARTBEAT)

    def _check(self):
        if self.error:
            raise ConnectionError(f"Wi-Fi link lost: {self.error}")

    def write(self, data):
        with self._send_lock:
            self.sock.sendall(data)

    def read(self, _size):
        self._check()
        with self._buf_lock:
            end = self._buf.rfind(b"}") + 1
            out = bytes(self._buf[:end])
            del self._buf[:end]
        if not out:
            time.sleep(0.02)
        return out

    def reset_input_buffer(self):
        with self._buf_lock:
            end = self._buf.rfind(b"}") + 1
            del self._buf[:end]

    def close(self):
        self._alive = False
        self._pump.join(timeout=1)
        self.sock.close()


class Robot:
    def __init__(self, link, reply_timeout=0.6):
        self.link = link
        self.reply_timeout = reply_timeout
        self._id = 0

    @classmethod
    def usb(cls, port=DEFAULT_PORT, baud=9600, boot_wait=2.5):
        ser = serial.Serial(port, baud, timeout=0.05)
        time.sleep(boot_wait)
        ser.reset_input_buffer()
        return cls(ser)

    @classmethod
    def wifi(cls, host=DEFAULT_HOST, port=100):
        return cls(WifiLink(host, port), reply_timeout=5.0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        try:
            self.stop()
        finally:
            self.link.close()

    def _request(self, payload):
        self._id = self._id % 9000 + 1
        hid = str(self._id)
        msg = json.dumps({"H": hid, **payload}, separators=(",", ":")).encode()
        self.link.reset_input_buffer()
        self.link.write(msg)
        buf = b""
        deadline = time.monotonic() + self.reply_timeout
        while time.monotonic() < deadline:
            buf += self.link.read(64)
            for m in REPLY.finditer(buf):
                body = m.group(1).decode(errors="replace")
                if body.startswith(hid + "_"):
                    return body[len(hid) + 1:]
        raise TimeoutError(f"no reply to {msg.decode()}; got {buf!r}")

    def stop(self):
        self._request({"N": 110})  # clears every running function and, unlike N=100, is acknowledged

    def drive(self, direction, speed):
        self._request({"N": 3, "D1": DIRECTIONS[direction], "D2": max(0, min(255, int(speed)))})

    def move(self, direction, speed, ms):
        self._request({"N": 2, "D1": DIRECTIONS[direction], "D2": max(0, min(255, int(speed))), "T": int(ms)})

    def _status(self, reply):
        yaw, grounded = reply.split(",")
        return int(yaw) / 10.0, grounded == "1"  # heading grows to the left, checked against the camera

    def wheels(self, left, right):
        """Signed wheel speeds, -255..255. The firmware stops them 0.4 s after the last command.

        Returns (heading in degrees, on ground).
        """
        # The firmware's "left" motor channel drives the car's right wheels, checked against the camera.
        return self._status(self._request({"N": 4, "D1": int(right), "D2": int(left)}))

    def heading(self):
        return self._status(self._request({"N": 24}))

    def head(self, angle):
        self._request({"N": 5, "D1": 1, "D2": max(0, min(180, int(angle)))})

    def distance(self):
        cm = int(self._request({"N": 21, "D1": 2}))
        return MAX_RANGE_CM if cm <= 0 or cm > MAX_RANGE_CM else cm

    def line(self):
        return tuple(int(self._request({"N": 22, "D1": i})) for i in range(3))

    def on_ground(self):
        return self._request({"N": 23}) == "false"  # N=23 answers "has the car left the ground?"
