"""Browser control pane for the car agent: live view of camera, sensors and agent, plus manual control.

agent.py starts it on http://localhost:8765. The car takes a single control connection, so the pane
shares the agent's link: every sensor read and command goes through Car and its lock.
"""

import asyncio
import collections
import json
import logging
import time
from pathlib import Path

from aiohttp import WSMsgType, web

log = logging.getLogger("car-dashboard")

PAGE = Path(__file__).with_name("dashboard.html")
HEAD_CENTER = 90
HEAD_MIN, HEAD_MAX = 10, 170
HEAD_STEP = 15  # each servo command takes ~0.5 s on the car, so small steps crawl
SONAR_KEEP_S = 8.0
POLL_S = 0.12
WATCHDOG_S = 0.4  # manual drive stops when the browser stops confirming the key is held
SAFETY_S = 0.12

# Measured from the CAD model (sim/urdf/elegoo_v4.urdf) and the firmware; drawn to scale by the pane.
GEOMETRY = {
    "body_length_cm": 23.2,
    "body_width_cm": 13.2,
    "nose_x_cm": 11.5,
    "wheel_x_cm": 4.1,
    "wheel_y_cm": 5.8,
    "wheel_radius_cm": 3.33,
    "wheel_width_cm": 2.6,
    "sonar_x_cm": 11.0,
    "sonar_height_cm": 10.2,
    "sonar_cone_deg": 15,
    "sonar_max_cm": 150,
    "camera_x_cm": 8.3,
    "camera_height_cm": 12.6,
    "camera_hfov_deg": 52,
    "camera_vfov_deg": 40,
    "camera_tilt_deg": 10,  # estimate: the camera points up and misses the floor close ahead
    "line_x_cm": 9.2,
    "line_threshold": 950,
}


class Telemetry:
    def __init__(self, stop_cm):
        self.version = 0
        self.link = {"state": "connecting", "rtt_ms": None, "error": None}
        self.sensors = {"distance": None, "distance_t": None, "head": HEAD_CENTER, "line": None, "on_ground": None}
        self.sonar = collections.deque(maxlen=120)
        self.motion = {"by": None, "dir": None}
        self.frame = {"id": 0, "t": None, "jpeg": None}
        self.agent = {"state": "initializing", "user_state": "listening", "autonomy": True, "mic": True}
        self.transcript = collections.deque(maxlen=40)
        self.tools = collections.deque(maxlen=24)
        self.notices = collections.deque(maxlen=10)
        self.stop_cm = stop_cm
        self._ids = 0

    def touch(self):
        self.version += 1

    def next_id(self):
        self._ids += 1
        return self._ids

    def set_frame(self, jpeg):
        self.frame = {"id": self.frame["id"] + 1, "t": time.time(), "jpeg": jpeg}
        self.touch()

    def add_distance(self, head, cm):
        now = time.time()
        self.sensors.update(distance=cm, distance_t=now)
        self.sonar.append((head, cm, now))
        self.touch()

    def moved(self, by, direction):
        if direction is not None:
            self.sonar.clear()  # readings are relative to the car, so they go stale once it moves
        self.motion = {"by": by, "dir": direction}
        self.touch()

    def say(self, role, text, final=True):
        last = self.transcript[-1] if self.transcript else None
        if role == "user" and last and last["role"] == "user" and (not last["final"] or last["text"] == text):
            last.update(text=text, final=final, t=time.time())
        else:
            self.transcript.append({"id": self.next_id(), "role": role, "text": text, "final": final, "t": time.time()})
        self.touch()

    def tool_start(self, label):
        entry = {"id": self.next_id(), "label": label, "status": "running", "result": None, "t0": time.time(), "t1": None}
        self.tools.append(entry)
        self.touch()
        return entry

    def tool_end(self, entry, result, ok=True):
        entry.update(status="done" if ok else "error", result=result, t1=time.time())
        self.touch()

    def notice(self, text, level="info"):
        self.notices.append({"id": self.next_id(), "text": text, "level": level, "t": time.time()})
        self.touch()

    def snapshot(self):
        now = time.time()
        return {
            "t": "state",
            "now": now,
            "link": self.link,
            "sensors": self.sensors,
            "sonar": [[h, cm, round(now - t, 2)] for h, cm, t in self.sonar if now - t < SONAR_KEEP_S],
            "motion": self.motion,
            "frame": {"id": self.frame["id"], "t": self.frame["t"]},
            "agent": self.agent,
            "transcript": list(self.transcript),
            "tools": list(self.tools),
            "notices": list(self.notices),
            "stop_cm": self.stop_cm,
        }


class Recorder:
    def __init__(self, robot, telemetry, by):
        self._robot = robot
        self._t = telemetry
        self._by = by

    def _timed(self, fn, *args):
        start = time.monotonic()
        result = fn(*args)
        self._t.link.update(state="online", rtt_ms=round((time.monotonic() - start) * 1000), error=None)
        return result

    def head(self, angle):
        angle = max(0, min(180, int(angle)))
        self._timed(self._robot.head, angle)
        self._t.sensors["head"] = angle
        self._t.touch()

    def distance(self):
        cm = self._timed(self._robot.distance)
        self._t.add_distance(self._t.sensors["head"], cm)
        return cm

    def line(self):
        values = self._timed(self._robot.line)
        self._t.sensors["line"] = list(values)
        self._t.touch()
        return values

    def on_ground(self):
        grounded = self._timed(self._robot.on_ground)
        self._t.sensors["on_ground"] = grounded
        self._t.touch()
        return grounded

    def drive(self, direction, speed):
        self._timed(self._robot.drive, direction, speed)
        self._t.moved(self._by, direction)

    def move(self, direction, speed, ms):
        self._t.moved(self._by, direction)
        try:
            self._timed(self._robot.move, direction, speed, ms)
        finally:
            self._t.moved(None, None)

    def wheels(self, left, right):
        if not (left or right):
            direction = None
        elif (left > 0) == (right > 0):
            direction = "forward" if left > 0 else "back"
        else:
            direction = "left" if right > left else "right"
        if direction != self._t.motion["dir"]:
            self._t.moved(self._by if direction else None, direction)
        return self._timed(self._robot.wheels, left, right)

    def stop(self):
        self._timed(self._robot.stop)
        self._t.moved(None, None)

    def __getattr__(self, name):
        return getattr(self._robot, name)


class Pilot:
    def __init__(self, car, telemetry, speed, scan):
        self.car = car
        self.t = telemetry
        self.speed = speed
        self._scan = scan
        self.drive_dir = None
        self.drive_seen = 0.0
        self.pan_dir = 0
        self.pan_seen = 0.0
        self._driving = None
        self._last_safety = 0.0
        self._last_pan = 0.0
        self._poll_i = 0
        self._busy = False

    async def _car(self, fn, by="human"):
        return await asyncio.to_thread(self.car.run_quiet, fn, by)

    def set_drive(self, direction):
        if direction is not None:
            self.car.human_input()
        self.drive_dir = direction
        self.drive_seen = time.monotonic()

    def set_pan(self, direction):
        if direction:
            self.car.human_input()
        self.pan_dir = direction
        self.pan_seen = time.monotonic()

    def heartbeat(self):
        now = time.monotonic()
        if self.drive_dir:
            self.drive_seen = now
            self.car.human_input()
        if self.pan_dir:
            self.pan_seen = now

    def release_all(self):
        self.drive_dir = None
        self.pan_dir = 0

    async def estop(self):
        self.release_all()
        self.car.human_input()
        await self._car(lambda r: r.stop())
        self._driving = None
        self.t.notice("Emergency stop", "warn")

    async def center(self):
        self.car.human_input()
        await self._car(lambda r: r.head(HEAD_CENTER))

    async def scan(self):
        self.car.human_input()
        self.t.notice("Scanning surroundings")
        await self._car(self._scan)

    def _blocked(self, r):
        if not r.on_ground():
            return "No floor under the front sensors"
        if abs(self.t.sensors["head"] - HEAD_CENTER) > 3:
            r.head(HEAD_CENTER)
        cm = r.distance()
        if cm < self.t.stop_cm:
            return f"Obstacle {cm} cm ahead"
        return None

    async def _drive_tick(self, now):
        want = self.drive_dir
        if want and now - self.drive_seen > WATCHDOG_S:
            self.drive_dir = want = None
            self.t.notice("Control signal lost, stopped", "warn")
        if want != self._driving:
            if want is None:
                await self._car(lambda r: r.stop())
                self._driving = None
                return
            if want == "forward":
                reason = await self._car(self._blocked)
                if reason:
                    self.drive_dir = None
                    self.t.notice(f"{reason}, not driving forward", "warn")
                    if self._driving:
                        await self._car(lambda r: r.stop())
                        self._driving = None
                    return
            await self._car(lambda r: r.drive(want, self.speed))
            self._driving = want
            self._last_safety = now
        elif self._driving == "forward" and now - self._last_safety > SAFETY_S:
            self._last_safety = now
            reason = await self._car(self._blocked)
            if reason:
                await self._car(lambda r: r.stop())
                self._driving = self.drive_dir = None
                self.t.notice(f"{reason}, stopped", "warn")

    async def _pan_tick(self, now):
        if self.pan_dir and now - self.pan_seen > WATCHDOG_S:
            self.pan_dir = 0
        if self.pan_dir and now - self._last_pan > 0.08:
            self._last_pan = now
            target = max(HEAD_MIN, min(HEAD_MAX, self.t.sensors["head"] + HEAD_STEP * self.pan_dir))
            if target != self.t.sensors["head"]:
                await self._car(lambda r: r.head(target))

    def _poll(self, r):
        self._poll_i += 1
        if self._poll_i % 6 == 0:
            r.line()
        elif self._poll_i % 6 == 3:
            r.on_ground()
        else:
            r.distance()

    async def run(self):
        while True:
            now = time.monotonic()
            try:
                await self._drive_tick(now)
                await self._pan_tick(now)
                if self._driving is None and not self.pan_dir:
                    await asyncio.to_thread(self.car.try_run, self._poll)
            except ValueError as e:
                log.warning("pilot: unreadable reply from the car: %s", e)
            await asyncio.sleep(POLL_S if self._driving is None and not self.pan_dir else 0.04)


class Dashboard:
    def __init__(self, telemetry, pilot, actions, stream_url, port=8765):
        self.t = telemetry
        self.pilot = pilot
        self.actions = actions
        self.stream_url = stream_url
        self.port = port
        self._clients = set()
        self._runner = None
        self._tasks = []

    async def start(self):
        app = web.Application()
        app.router.add_get("/", self._page)
        app.router.add_get("/frame.jpg", self._frame)
        app.router.add_get("/ws", self._ws)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", self.port, reuse_address=True).start()
        self._tasks = [asyncio.create_task(self.pilot.run()), asyncio.create_task(self._broadcast())]
        log.info("control pane: http://localhost:%d", self.port)

    async def close(self):
        for task in self._tasks:
            task.cancel()
        for ws in list(self._clients):
            await ws.close()
        if self._runner:
            await self._runner.cleanup()

    async def _page(self, _request):
        return web.FileResponse(PAGE, headers={"Cache-Control": "no-store"})

    async def _frame(self, _request):
        jpeg = self.t.frame["jpeg"]
        if jpeg is None:
            return web.Response(status=404, text="no frame yet")
        return web.Response(body=jpeg, content_type="image/jpeg", headers={"Cache-Control": "no-store"})

    async def _broadcast(self):
        sent = -1
        while True:
            await asyncio.sleep(0.1)
            if self._clients and self.t.version != sent:
                sent = self.t.version
                payload = json.dumps(self.t.snapshot())
                for ws in list(self._clients):
                    try:
                        await ws.send_str(payload)
                    except ConnectionError:
                        self._clients.discard(ws)

    async def _ws(self, request):
        ws = web.WebSocketResponse(heartbeat=5)
        await ws.prepare(request)
        self._clients.add(ws)
        await ws.send_str(json.dumps({"t": "hello", "geometry": GEOMETRY, "stream_url": self.stream_url, "speed": self.pilot.speed}))
        await ws.send_str(json.dumps(self.t.snapshot()))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await self._command(json.loads(msg.data))
        finally:
            self._clients.discard(ws)
            if not self._clients:
                self.pilot.release_all()
        return ws

    async def _command(self, cmd):
        kind = cmd.get("t")
        if kind == "drive":
            self.pilot.set_drive(cmd.get("dir"))
        elif kind == "pan":
            self.pilot.set_pan(int(cmd.get("dir", 0)))
        elif kind == "hb":
            self.pilot.heartbeat()
        elif kind == "center":
            await self.pilot.center()
        elif kind == "scan":
            asyncio.create_task(self.pilot.scan())
        elif kind == "estop":
            await self.pilot.estop()
            await self.actions.estop()
        elif kind == "speed":
            self.pilot.speed = max(60, min(255, int(cmd.get("value", self.pilot.speed))))
        elif kind == "autonomy":
            self.t.agent["autonomy"] = bool(cmd.get("on"))
            self.t.notice("Agent driving enabled" if self.t.agent["autonomy"] else "Agent driving paused")
        elif kind == "mic":
            self.actions.set_mic(bool(cmd.get("on")))
        elif kind == "say" and cmd.get("text", "").strip():
            self.actions.say(cmd["text"].strip())
        self.t.touch()
