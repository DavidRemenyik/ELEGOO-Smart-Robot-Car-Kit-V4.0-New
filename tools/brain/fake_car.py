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
import socket
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
PERSON_RADIUS = 17

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
        self.person = None
        self.lifted = False
        self.drop_until = 0.0
        self.zero_at = 0.0
        self.barrier = None  # x0, y0, x1, y1: stops the car, but neither the range sensor nor the camera sees it
        self.view = None  # "blank" (a plain surface fills the view) or "dark" replaces the rendered picture

    def _set_wheels(self, left, right, source):
        if (self.left or self.right) and not (left or right):
            self.zero_at = time.monotonic()
            log(f"WHEELS ZERO L=0 R=0 source={source} monotonic={self.zero_at:.6f}")
        self.left, self.right = left, right

    def _advance(self):
        now = time.monotonic()
        while self.t < now:
            dt = min(0.01, now - self.t)
            if self.t >= self.last_wheels + SELF_STOP_S:
                self._set_wheels(0, 0, "watchdog")
            vl, vr = wheel_speed(self.left), wheel_speed(self.right)
            if vl or vr:
                a = math.radians(self.heading)
                v = (vl + vr) / 2
                nx, ny = self.x + v * math.cos(a) * dt, self.y + v * math.sin(a) * dt
                if self.lifted or blocked(nx, ny) or (
                    self.person and math.hypot(nx - self.person[0], ny - self.person[1]) < PERSON_RADIUS + CAR_RADIUS
                ) or (self.barrier and self.barrier[0] - CAR_RADIUS < nx < self.barrier[2] + CAR_RADIUS
                      and self.barrier[1] - CAR_RADIUS < ny < self.barrier[3] + CAR_RADIUS):
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
        return f"{round(self.heading * 10)},{int(not self.lifted)}"

    def test_state(self):
        with self.lock:
            self._advance()
            return dict(x=self.x, y=self.y, heading=self.heading, head=self.head, left=self.left,
                        right=self.right, person=self.person, lifted=self.lifted, barrier=self.barrier,
                        view=self.view, zero_at=self.zero_at, now=time.monotonic())

    def handle(self, msg):
        """Apply one command; returns the reply body."""
        n = msg.get("N")
        with self.lock:
            self._advance()
            if n == 4:
                # robot.py sends D1 = right wheels, D2 = left wheels
                right = max(-255, min(255, int(msg.get("D1", 0))))
                left = max(-255, min(255, int(msg.get("D2", 0))))
                self._set_wheels(left, right, "N=4")
                self.last_wheels = time.monotonic()
                return self.status()
            if n == 3:
                speed = max(0, min(255, int(msg.get("D2", 0))))
                left, right = {1: (speed, -speed), 2: (-speed, speed), 3: (speed, speed),
                               4: (-speed, -speed)}.get(msg.get("D1"), (0, 0))
                self._set_wheels(left, right, "manual")
                self.last_wheels = math.inf  # N=3 stays active until an explicit stop
                return "ok"
            if n == 24:
                return self.status()
            if n == 21:
                return str(round(self._ahead()))
            if n == 22:
                return "1000" if self.lifted else "120"
            if n == 23:
                return "true" if self.lifted else "false"
            if n == 5:
                self.head = max(0, min(180, int(msg.get("D2", CENTER))))
                return "ok"
            if n in (110, 100):
                self._set_wheels(0, 0, f"N={n}")
                return "ok"
            return "ok"

    def _ahead(self):
        ang = math.radians(self.heading + self.head - CENTER)
        d, _ = ray_walls(self.x, self.y, ang)
        b = ray_box(self.x, self.y, ang)
        if b is not None:
            d = min(d, b)
        if self.person:
            dx, dy = self.person[0] - self.x, self.person[1] - self.y
            offset = (math.atan2(dy, dx) - ang + math.pi) % math.tau - math.pi
            if abs(offset) < math.radians(13):
                d = min(d, math.hypot(dx, dy) - PERSON_RADIUS)
        return max(2.0, d - 5.0)  # the sensor sits about 5 cm behind the bumper line it sees from


def render(car):
    scene = car.test_state()
    if scene["view"]:
        level = 8 if scene["view"] == "dark" else 150
        noise = Image.effect_noise((W, H), 3).point(lambda v: max(0, min(255, level + (v - 128))))
        out = io.BytesIO()
        noise.convert("RGB").save(out, "JPEG", quality=80)
        return out.getvalue(), ""
    x, y, heading, head = (scene[k] for k in ("x", "y", "heading", "head"))
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
    fixture = ""
    if scene["person"]:
        px, py = scene["person"]
        dx, dy = px - x, py - y
        z = dx * math.cos(view) + dy * math.sin(view)
        if z > 10:
            cx = W / 2 + FX * (dx * math.sin(view) - dy * math.cos(view)) / z
            half = FX * PERSON_RADIUS / z
            top = HORIZON - FY * (165 - CAM_H) / z
            base = HORIZON + FY * CAM_H / z
            x0, x1 = max(0, cx - half), min(W, cx + half)
            y0, y1 = max(0, top), min(H, base)
            if x1 - x0 > 5 and y1 - y0 > 12:
                height = base - top
                draw.ellipse((cx - half / 2, top, cx + half / 2, top + 0.18 * height), fill="#d8a581")
                draw.rounded_rectangle((cx - half, top + 0.17 * height, cx + half, top + 0.60 * height),
                                       radius=int(half / 4), fill="#355cc3")
                for side in (-1, 1):
                    mid = cx + side * 0.53 * half
                    draw.rectangle((mid - 0.32 * half, top + 0.58 * height, mid + 0.32 * half, base),
                                   fill="#202030")
                fixture = ",".join(f"{v:.5f}" for v in (x0 / W, y0 / H, x1 / W, y1 / H))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=80)
    return out.getvalue(), fixture


def camera_server(car, port, test_controls):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _state(self):
            data = json.dumps(car.test_state()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if not test_controls or self.path != "/test":
                self.send_error(404)
                return
            try:
                size = int(self.headers["Content-Length"])
                if not 0 < size <= 1024:
                    raise ValueError("wrong request size")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict) or not data or data.keys() - {
                    "person", "lifted", "drop_link", "pose", "barrier", "view"
                }:
                    raise ValueError("unknown test command")
                barrier = data.get("barrier")
                if barrier is not None:
                    x0, y0, x1, y1 = map(float, barrier)
                    if not (x0 < x1 and y0 < y1):
                        raise ValueError("barrier needs x0 < x1 and y0 < y1")
                    barrier = (x0, y0, x1, y1)
                if data.get("view") not in (None, "blank", "dark"):
                    raise ValueError("view is blank, dark or null")
                pose = None
                if "pose" in data:
                    values = data["pose"]
                    if not isinstance(values, list) or len(values) != 3 or any(
                        type(value) not in (int, float) or not math.isfinite(value) for value in values
                    ):
                        raise ValueError("pose needs three finite numbers")
                    x, y, heading = values
                    if blocked(x, y):
                        raise ValueError("pose intersects a wall or the box")
                    pose = (x, y, heading)
                with car.lock:
                    car._advance()
                    if pose is not None:
                        car._set_wheels(0, 0, "pose reset")
                        car.x, car.y, car.heading = pose
                        car.head = CENTER
                        car.bumped = False
                    if "person" in data:
                        person = data["person"]
                        if person is not None:
                            px, py = map(float, person)
                            if not (0 <= px <= ROOM_W and 0 <= py <= ROOM_H):
                                raise ValueError("person outside room")
                            person = (px, py)
                        car.person = person
                    if "lifted" in data:
                        car.lifted = data["lifted"] is True
                    if data.get("drop_link") is True:
                        car.drop_until = time.monotonic() + 1
                    if "barrier" in data:
                        car.barrier = barrier
                    if "view" in data:
                        car.view = data["view"]
                    log(f"FIXTURE {data} monotonic={time.monotonic():.6f}")
                self._state()
            except (KeyError, OverflowError, TypeError, ValueError):
                self.send_error(400)

        def do_GET(self):
            if self.path == "/test" and test_controls:
                self._state()
            elif self.path.startswith("/capture"):
                jpeg, fixture = render(car)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                if test_controls and fixture:
                    self.send_header("X-Fake-Person", fixture)
                self.end_headers()
                self.wfile.write(jpeg)
            elif self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        jpeg, _ = render(car)
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
            if car.drop_until > time.monotonic():
                break
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


async def physics(car, clients):
    """Advance the watchdog independently of incoming packets and camera requests."""
    while True:
        car.pose()
        if car.drop_until > time.monotonic():
            for writer in tuple(clients):
                writer.transport.abort()
        await asyncio.sleep(0.01)


async def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=18100)
    ap.add_argument("--camera-port", type=int, default=18101)
    ap.add_argument("--test-controls", action="store_true", help="enable loopback fixture controls at /test")
    args = ap.parse_args()
    car = Car()
    camera_server(car, args.camera_port, args.test_controls)
    clients = set()

    async def connected(reader, writer):
        clients.add(writer)
        try:
            await serve_client(car, reader, writer)
        finally:
            clients.discard(writer)

    server = await asyncio.start_server(connected, "127.0.0.1", args.port)
    log(f"fake car on 127.0.0.1:{args.port}, camera http://127.0.0.1:{args.camera_port}/capture and /stream")
    async with server, asyncio.TaskGroup() as group:
        group.create_task(physics(car, clients))
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
