"""Stand-in for the car, for working on the Mac without hardware.

    .venv/bin/python fake_car.py --port 18100 --camera-port 18101

Speaks the car's JSON-over-TCP protocol ({"H":id,"N":cmd,...} answered by {id_reply}) with a
45 ms reply delay, sends {Heartbeat} and expects it echoed like the firmware, and moves a simulated
car in a rectangular room: N=4 wheel speeds turn into heading and position, and the wheels stop by
themselves 0.4 s after the last N=4. N=21 measures the distance to whatever is ahead of the head.
The camera port serves /capture (one JPEG rendered from the car's pose, camera upright and tilted
up 20 degrees like the real one) and /stream (MJPEG). Every command is logged to stdout.

Room: 600 x 400 cm, x to the east, y to the north, heading 0 = facing east, positive = left (counter-
clockwise seen from above), as the real gyro reports it. Walls: north red, east green, south blue,
west yellow, with a brown door in the east wall; an orange box stands in the room.
"""

import argparse
import asyncio
import io
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, ImageDraw, ImageFont

ROOM_W, ROOM_H = 600.0, 400.0
WALL_CM = 250.0
DOOR_Y = (160.0, 250.0)  # span along the east wall
DOOR_CM = 200.0
BOX = (420.0, 280.0, 480.0, 340.0)  # x0, y0, x1, y1; 40 cm tall
BOX_TALL = 40.0
CAR_RADIUS = 12.0
START = (150.0, 200.0, 0.0)  # x, y, heading

REPLY_DELAY_S = 0.045
SELF_STOP_S = 0.4
HEARTBEAT_S = 1.0
CM_PER_S, CRUISE, DEAD = 55.0, 150, 12  # same speed line as drive.py's default calibration
TRACK_CM = 24.0  # effective, so PWM 80 spins about 130 deg/s like the real car
CENTER = 90

CAM_H = 15.0
CAM_PITCH = math.radians(20)
W, H = 640, 480
FX = (W / 2) / math.tan(math.radians(30))  # 60 degree horizontal view
FY = (H / 2) / math.tan(math.radians(24))  # 48 degree vertical view
HORIZON = H / 2 + FY * math.tan(CAM_PITCH)  # looking up moves the horizon down the picture
COLUMNS = 160
WALLS = {"north": (200, 40, 40), "east": (40, 160, 60), "south": (50, 80, 200), "west": (220, 190, 40)}
BOX_COLOR = (240, 130, 20)
DOOR_COLOR = (90, 55, 25)
CEILING_COLOR = (205, 205, 210)
FLOOR_COLOR = (120, 90, 60)


def wheel_speed(pwm):
    return math.copysign(CM_PER_S * max(0.0, abs(pwm) - DEAD) / (CRUISE - DEAD), pwm)


def ray_walls(x, y, ang):
    """Distance along a ray from inside the room to the first wall, and that wall's name."""
    dx, dy = math.cos(ang), math.sin(ang)
    hits = []
    if dx > 1e-9:
        hits.append(((ROOM_W - x) / dx, "east"))
    if dx < -1e-9:
        hits.append((-x / dx, "west"))
    if dy > 1e-9:
        hits.append(((ROOM_H - y) / dy, "north"))
    if dy < -1e-9:
        hits.append((-y / dy, "south"))
    return min(hits)


def ray_box(x, y, ang):
    """Distance along a ray to the box, or None."""
    dx, dy = math.cos(ang), math.sin(ang)
    near, far = -math.inf, math.inf
    for o, d, lo, hi in ((x, dx, BOX[0], BOX[2]), (y, dy, BOX[1], BOX[3])):
        if abs(d) < 1e-9:
            if not lo <= o <= hi:
                return None
            continue
        t0, t1 = sorted(((lo - o) / d, (hi - o) / d))
        near, far = max(near, t0), min(far, t1)
    return near if near <= far and near > 0 else None


def blocked(x, y):
    r = CAR_RADIUS
    if not (r <= x <= ROOM_W - r and r <= y <= ROOM_H - r):
        return True
    return BOX[0] - r < x < BOX[2] + r and BOX[1] - r < y < BOX[3] + r


class Car:
    def __init__(self):
        self.lock = threading.Lock()
        self.x, self.y, self.heading = START
        self.left = self.right = 0
        self.head = CENTER
        self.last_wheels = 0.0
        self.t = time.monotonic()
        self.bumped = False

    def _advance(self):
        now = time.monotonic()
        while self.t < now:
            dt = min(0.01, now - self.t)
            if self.t >= self.last_wheels + SELF_STOP_S:
                self.left = self.right = 0
            vl, vr = wheel_speed(self.left), wheel_speed(self.right)
            if vl or vr:
                a = math.radians(self.heading)
                v = (vl + vr) / 2
                nx, ny = self.x + v * math.cos(a) * dt, self.y + v * math.sin(a) * dt
                if blocked(nx, ny):
                    self.bumped = True
                else:
                    self.x, self.y = nx, ny
                self.heading += math.degrees((vr - vl) / TRACK_CM * dt)
            self.t += dt

    def pose(self):
        with self.lock:
            self._advance()
            return self.x, self.y, self.heading, self.head

    def status(self):
        return f"{round(self.heading * 10)},1"

    def handle(self, msg):
        """Apply one command; returns the reply body."""
        n = msg.get("N")
        with self.lock:
            self._advance()
            if n == 4:
                # robot.py sends D1 = right wheels, D2 = left wheels
                self.right = max(-255, min(255, int(msg.get("D1", 0))))
                self.left = max(-255, min(255, int(msg.get("D2", 0))))
                self.last_wheels = time.monotonic()
                return self.status()
            if n == 24:
                return self.status()
            if n == 21:
                return str(round(self._ahead()))
            if n == 22:
                return "120"  # bright floor under every line sensor
            if n == 23:
                return "false"  # not lifted
            if n == 5:
                self.head = max(0, min(180, int(msg.get("D2", CENTER))))
                return "ok"
            if n in (110, 100):
                self.left = self.right = 0
                return "ok"
            return "ok"

    def _ahead(self):
        ang = math.radians(self.heading + self.head - CENTER)
        d, _ = ray_walls(self.x, self.y, ang)
        b = ray_box(self.x, self.y, ang)
        if b is not None:
            d = min(d, b)
        return max(2.0, d - 5.0)  # the sensor sits about 5 cm behind the bumper line it sees from


def render(car):
    x, y, heading, head = car.pose()
    view = math.radians(heading + head - CENTER)
    img = Image.new("RGB", (W, H), CEILING_COLOR)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, int(HORIZON), W, H], fill=FLOOR_COLOR)
    seen = {}
    step = W // COLUMNS
    for c in range(COLUMNS):
        off = math.atan(((c + 0.5) * step - W / 2) / FX)
        ang = view - off  # picture x grows to the right, angles grow to the left
        d, wall = ray_walls(x, y, ang)
        z = max(1.0, d * math.cos(off))
        shade = 0.55 + 0.45 * min(1.0, 120 / z)
        color = tuple(int(v * shade) for v in WALLS[wall])
        top, bottom = HORIZON - FY * (WALL_CM - CAM_H) / z, HORIZON + FY * CAM_H / z
        draw.rectangle([c * step, int(top), c * step + step - 1, int(bottom)], fill=color)
        seen[wall] = min(seen.get(wall, math.inf), d)
        if wall == "east" and DOOR_Y[0] <= y + d * math.sin(ang) <= DOOR_Y[1]:
            draw.rectangle([c * step, int(HORIZON - FY * (DOOR_CM - CAM_H) / z), c * step + step - 1, int(bottom)], fill=DOOR_COLOR)
            seen["door"] = min(seen.get("door", math.inf), d)
        b = ray_box(x, y, ang)
        if b is not None and b < d:
            zb = max(1.0, b * math.cos(off))
            draw.rectangle([c * step, int(HORIZON - FY * (BOX_TALL - CAM_H) / zb), c * step + step - 1, int(HORIZON + FY * CAM_H / zb)], fill=BOX_COLOR)
            seen["orange box"] = min(seen.get("orange box", math.inf), b)
    font = ImageFont.load_default(size=18)
    lines = ["FAKE CAR CAMERA (synthetic room)"] + [f"{name}{' wall' if name in WALLS else ''}: {d:.0f} cm" for name, d in sorted(seen.items(), key=lambda kv: kv[1])]
    lines.append(f"pose x={x:.0f} y={y:.0f} heading={heading:.0f} head={head}")
    draw.rectangle([0, 0, 360, 8 + 22 * len(lines)], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((8, 4 + 22 * i), line, fill=(255, 255, 255), font=font)
    out = io.BytesIO()
    img.save(out, "JPEG", quality=80)
    return out.getvalue()


def camera_server(car, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path.startswith("/capture"):
                jpeg = render(car)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
            elif self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        jpeg = render(car)
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                        time.sleep(0.2)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def split_messages(buf):
    """Pull complete {...} messages out of the buffer; returns (messages, rest)."""
    out = []
    while True:
        start = buf.find(b"{")
        end = buf.find(b"}", start)
        if start < 0 or end < 0:
            return out, buf[start:] if start >= 0 else b""
        out.append(buf[start:end + 1])
        buf = buf[end + 1:]


def log(text):
    print(f"{time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d} {text}", flush=True)


async def serve_client(car, reader, writer):
    peer = writer.get_extra_info("peername")
    log(f"client {peer} connected")
    beats = {"sent": 0, "echoed": 0}

    async def heartbeat():
        while True:
            writer.write(b"{Heartbeat}")
            beats["sent"] += 1
            await writer.drain()
            await asyncio.sleep(HEARTBEAT_S)

    pulse = asyncio.create_task(heartbeat())
    buf = b""
    try:
        while chunk := await reader.read(1024):
            msgs, buf = split_messages(buf + chunk)
            for raw in msgs:
                if raw == b"{Heartbeat}":
                    beats["echoed"] += 1
                    continue
                try:
                    msg = json.loads(raw)
                except ValueError:
                    log(f"bad message {raw!r}")
                    continue
                reply = car.handle(msg)
                await asyncio.sleep(REPLY_DELAY_S)
                writer.write(b"{" + f"{msg.get('H')}_{reply}".encode() + b"}")
                await writer.drain()
                x, y, heading, head = car.pose()
                args = " ".join(f"{k}={v}" for k, v in msg.items() if k not in ("H", "N"))
                wheels = f" L={msg.get('D2')} R={msg.get('D1')}" if msg.get("N") == 4 else ""
                log(f"N={msg.get('N')} {args}{wheels} -> {reply}  pose x={x:.1f} y={y:.1f} heading={heading:.1f} head={head}")
    except (ConnectionError, OSError) as e:
        log(f"client {peer} error: {e}")
    finally:
        pulse.cancel()
        writer.close()
        log(f"client {peer} gone (heartbeats sent {beats['sent']}, echoed {beats['echoed']})")


async def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=18100)
    ap.add_argument("--camera-port", type=int, default=18101)
    args = ap.parse_args()
    car = Car()
    camera_server(car, args.camera_port)
    server = await asyncio.start_server(lambda r, w: serve_client(car, r, w), "127.0.0.1", args.port)
    log(f"fake car on 127.0.0.1:{args.port}, camera http://127.0.0.1:{args.camera_port}/capture and /stream")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
