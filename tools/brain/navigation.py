"""Go to a named place: the navigator of the project's sim/ folder, run live on the car's sensors.

This is the only brain module that imports simulator-side code. The project root (four folders up, the
folder holding sim/) is appended to sys.path once, here, because sim/navigate, sim/localize and
sim/realdepth import each other as `sim.<package>`. Their dependencies are in requirements.txt: scipy,
pyyaml, opencv and embreex everywhere, torch and depth_pro_2 (Apple Depth Pro 2-S) on the Mac only.

Observation, built each 100 ms control step (Sensors): depth from the newest Vision frame through
sim/realdepth/realdepth.py's observe(), on its own thread (DepthFeed); range from the head's sensor;
gyro = the heading in the wheel replies, unwrapped; head angle as commanded; wheel commands as sent.
Depth Pro 2 is under Apple's research-only licence; observation_spec.md names Depth Anything V2
(Apache-2.0, Core ML) as the fallback if that is unacceptable. Not wired in: realdepth.py has no DA2 path yet.

Test only: CAR_TEST_DEPTH_URL, a 127.0.0.1 URL of the stand-in's /depth (sim/macsim/standin.py
--test-depth), replaces realdepth with the simulator's own corrupted depth observation.

CAR_NAV_DRIVER: hand (default) | policy | policy_veto, see .omo/waves/p3/policy-nav.md (the learned policy is
faster but touched things 4.5 times as often per metre).
"""

import io
import logging
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from math import degrees, radians
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sim.localize.localise import Localizer  # noqa: E402
from sim.macsim.simulator import Observation  # noqa: E402
from sim.macsim.world import floor  # noqa: E402
from sim.navigate.navigator import DRIVERS, DT, Navigator  # noqa: E402
from sim.navigate.places import PLACES, area_name, places  # noqa: E402

log = logging.getLogger("car-agent")

DRIVER = os.getenv("CAR_NAV_DRIVER", "hand")
if DRIVER not in DRIVERS:
    raise SystemExit(f"CAR_NAV_DRIVER must be one of {', '.join(DRIVERS)}, not {DRIVER!r}")
TEST_DEPTH_URL = os.getenv("CAR_TEST_DEPTH_URL") or None
if TEST_DEPTH_URL and urlsplit(TEST_DEPTH_URL).hostname not in ("127.0.0.1", "::1"):
    raise SystemExit("CAR_TEST_DEPTH_URL is for the local stand-in only (127.0.0.1)")
DEPTH_MAX_AGE_S = 0.5  # an older depth frame is not passed to the navigator
FIRST_DEPTH_S = 30.0  # go_to waits this long for the first depth frame (the model loads on the first trip)
TRUST = 0.5  # the localiser's confidence from which its pose is trusted (localize.md)
# Saving home needs more: a wrong home spoils every later trip. localize.md: confidence > 0.7 while > 1 m off
# in at most 0.2 % of steps. A head pan alone locked the lounge 1.5 m off at 0.93, so locate also turns a circle.
HOME_TRUST = 0.8
PULSE_DEG, PULSE_STEPS, REST_STEPS, PAN_STEPS = 40, 8, 8, 25  # the navigator's relocalisation turn


class UnknownPlace(ValueError):
    """The name matches no place in places.yaml; the message lists the known ones."""


@dataclass(frozen=True, slots=True)
class Home:
    x: float
    y: float
    yaw: float
    area: str
    placeholder: bool


def home() -> Home:
    h = yaml.safe_load(PLACES.read_text())["home"]
    return Home(float(h["x"]), float(h["y"]), float(h["yaw"]), str(h["area"]), bool(h.get("placeholder")))


def save_home(pose: tuple[float, float, float]) -> Home:
    """Rewrite places.yaml's home line (the comments and the areas stay as they are)."""
    x, y, yaw = pose
    area = floor().area(x, y)
    if not area:
        raise ValueError("that position is outside every room on my map")
    line = f"home: {{x: {x:.2f}, y: {y:.2f}, yaw: {yaw:.4f}, area: {area}, placeholder: false}}"
    text, n = re.subn(r"^home:.*$", line, PLACES.read_text(), count=1, flags=re.MULTILINE)
    if n != 1:
        raise ValueError(f"no home line in {PLACES}")
    tmp = PLACES.with_suffix(".yaml.tmp")
    tmp.write_text(text)
    tmp.replace(PLACES)
    return home()


def place_names() -> list[str]:
    areas = yaml.safe_load(PLACES.read_text())["areas"].values()
    return [spec["name"] for spec in areas] + ["home"]


def find_place(name: str) -> tuple[str, tuple[float, float], str]:
    """Case-insensitive place name -> (name, goal x/y, area id)."""
    key = " ".join(name.casefold().split()).removeprefix("the ")
    if key == "home":
        h = home()
        return "home", (h.x, h.y), h.area
    if key in places():
        x, y, area = places()[key]
        return key, (x, y), area
    raise UnknownPlace(f"I don't know a place called {name.strip()!r}. I know: {', '.join(place_names())}.")


def room_name(pose: tuple[float, float, float]) -> str:
    area = floor().area(pose[0], pose[1]) if np.all(np.isfinite(pose[:2])) else ""
    return f"the {area_name(area)}" if area else "a spot I can't place on my map"


@dataclass(frozen=True, slots=True)
class Fix:
    pose: tuple[float, float, float]
    confidence: float
    at: float
    doing: str
    live: bool


class Whereabouts:
    """The localiser's latest guess, live while a go_to job runs, else the last one.

    It is trusted as the next trip's start only after a confident arrival (or a set home), with no wheel
    command from another job and no hand driving since.
    """

    def __init__(self, human_epoch: Callable[[], int]):
        self.human_epoch = human_epoch
        self.fix: Fix | None = None
        self._trusted_epoch: int | None = None
        self.last_trip: dict = {}

    def update(self, pose, confidence: float, doing: str, live: bool = True) -> None:
        self.fix = Fix(tuple(map(float, pose)), float(confidence), time.monotonic(), doing, live)

    def settle(self, trusted: bool) -> None:
        if self.fix:
            self.fix = Fix(self.fix.pose, self.fix.confidence, self.fix.at, self.fix.doing, False)
        self._trusted_epoch = self.human_epoch() if trusted and self.fix else None

    def moved(self) -> None:
        self._trusted_epoch = None

    def start_pose(self) -> tuple[float, float, float] | None:
        if self.fix and self._trusted_epoch is not None and self._trusted_epoch == self.human_epoch():
            return self.fix.pose
        return None

    def describe(self) -> str:
        fix = self.fix
        if fix is None or not np.all(np.isfinite(fix.pose)):
            return ("I don't know where I am yet. I work it out while going to a place, starting from home "
                    "or from a room you tell me.")
        x, y, _ = fix.pose
        guess = (f"I think I'm in {room_name(fix.pose)}" if fix.confidence >= TRUST
                 else f"My best guess is {room_name(fix.pose)}, but I'm not sure")
        detail = f"(about {x:.1f}, {y:.1f} on my map, confidence {fix.confidence:.0%})"
        if fix.live:
            return f"While {fix.doing}: {guess} {detail}."
        stale = "" if self.start_pose() else " I have moved or been moved since, so it may be out of date."
        return f"{guess} {detail}, as of {time.monotonic() - fix.at:.0f} seconds ago, when I was {fix.doing}.{stale}"


@dataclass(slots=True)
class Trip:
    label: str
    goal: tuple[float, float]
    navigator: Navigator
    start: str

    @property
    def there(self) -> str:
        return "home" if self.label == "home" else f"the {self.label}"

    @property
    def doing(self) -> str:
        return "going home" if self.label == "home" else f"going to the {self.label}"


def plan_trip(place: str, start_room: str | None, where: Whereabouts) -> Trip:
    """Pick the goal and the start prior: a named room, else where the last trip arrived, else home.

    Raises UnknownPlace for names not in places.yaml, ValueError for goals the map cannot reach.
    """
    label, goal, _ = find_place(place)
    if start_room:
        room, _, area = find_place(start_room)
        if room == "home":
            start_room = None
        else:
            return Trip(label, goal, Navigator(goal, area, driver=DRIVER), f"somewhere in the {room}")
    if not start_room and (pose := where.start_pose()):
        start, how = pose, "where my last trip ended"
    else:
        h = home()
        start, how = (h.x, h.y, h.yaw), "home" + (" (a placeholder spot until home is set)" if h.placeholder else "")
    return Trip(label, goal, Navigator(goal, start_pose=start, driver=DRIVER), how)


def _decode(jpeg: bytes) -> np.ndarray:
    import cv2

    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("camera frame is not a JPEG")
    if bgr.shape[:2] != (480, 640):  # realdepth's camera model is the car's 640x480 frame
        bgr = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class DepthFeed:
    """The newest depth observation, made on its own thread so the control loop never waits for it."""

    def __init__(self, vision):
        self.vision = vision
        self.problem: str | None = None
        self.ms: list[float] = []  # time per observation (decode + model), real path only
        self._latest: tuple[float, np.ndarray] | None = None
        self._taken = 0.0
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="nav-depth")
        self._thread.start()

    def _run(self) -> None:
        try:
            self._test() if TEST_DEPTH_URL else self._real()
        except Exception as e:  # noqa: BLE001 - any failure (model, import, link) means no depth; go_to reports it
            log.exception("Depth feed failed")
            self.problem = f"my depth camera pipeline failed ({e})"

    def _real(self) -> None:
        from sim.realdepth import realdepth  # torch + Depth Pro 2: Mac only, loaded on the first trip

        last = 0.0
        while not self._closed.is_set():
            frame = self.vision.newest(DEPTH_MAX_AGE_S)
            if frame is None or frame.timestamp <= last:
                self._closed.wait(0.01)
                continue
            last = frame.timestamp
            began = time.perf_counter()
            obs = realdepth.observe(_decode(frame.jpeg))
            self.ms.append((time.perf_counter() - began) * 1000)
            self._latest = (frame.timestamp, obs)

    def _test(self) -> None:
        while not self._closed.is_set():
            began = time.monotonic()
            with urlopen(TEST_DEPTH_URL, timeout=2) as response:
                body = response.read()
            if body:
                self._latest = (began, np.load(io.BytesIO(body)))
            self._closed.wait(max(0.0, DT - (time.monotonic() - began)))

    def ready(self) -> bool:
        return self._latest is not None

    def take(self) -> np.ndarray | None:
        """A frame not handed out before and younger than DEPTH_MAX_AGE_S, else None (never a replay)."""
        latest = self._latest
        if latest and latest[0] > self._taken and time.monotonic() - latest[0] < DEPTH_MAX_AGE_S:
            self._taken = latest[0]
            return latest[1]
        return None

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=1.0)


class Sensors:
    """Sends each step's command and reads the sensors for the next step's observation.

    The link answers one request at a time, about 57 ms each on the stand-in (45 ms reply, robot.py's 20 ms
    poll), and the navigator computes in about 7 ms. Two requests a step do not fit in 100 ms, so every step
    sends the wheel command (its reply carries the heading) and every second step also reads the range or
    moves the head; the other steps pass range None rather than an old reading.
    """

    def __init__(self, r, feed: DepthFeed):
        self.r, self.feed = r, feed
        r.head(90)
        self.prev_yaw, _ = r.heading()
        self.gyro = 0.0
        self.head = self.want_head = 0.0
        self.range: int | None = r.distance()
        self.sent = (0, 0)
        self.steps = 0

    def observe(self) -> Observation:
        return Observation(self.gyro, self.range, self.feed.take(), self.head, True)

    def send(self, left: int, right: int, head: float) -> None:
        self.steps += 1
        self.range = None
        if self.steps % 2 == 0:
            servo = max(0, min(180, round(90 + degrees(self.want_head))))
            if abs(servo - self.r.head_angle) >= 2:
                self.r.head(servo)
                self.head = radians(servo - 90)
            else:
                self.range = self.r.distance()
        command = (int(left), int(right))
        if command == (0, 0) and self.sent == (0, 0):
            yaw, _ = self.r.heading()  # the wheels already stand: only the heading is needed
        else:
            yaw, _ = self.r.wheels(*command)
        self.sent = command
        self.want_head = head
        self.gyro += radians((yaw - self.prev_yaw + 180) % 360 - 180)
        self.prev_yaw = yaw


def locate(r, feed: DepthFeed, area: str, where: Whereabouts, seconds: float = 50.0):
    """Find the car's pose in `area` on the spot: pan the head, turn one full circle in pulses, then hold still
    until the localiser has HOME_TRUST for 2 s. None if it never does within `seconds`."""
    loc = Localizer()
    loc.reset(room=area)
    sensors = Sensors(r, feed)
    sure, turned, step, pulse = 0, 0.0, 0, None  # pulse = (gyro at its start, steps so far) while turning
    due = time.monotonic()
    for _ in range(int(seconds / DT)):
        obs = sensors.observe()
        est = loc.update(sensors.sent, obs.gyro, obs.range_cm, obs.depth, obs.head_angle)
        where.update(est.pose, est.confidence, "finding my spot")
        step += 1
        command = (0, 0)
        if step > PAN_STEPS and turned < 2 * np.pi:
            if pulse is None:
                if step % REST_STEPS == 0:
                    pulse = (obs.gyro, 0)
            else:
                swept = abs(obs.gyro - pulse[0])
                if swept >= radians(PULSE_DEG) or pulse[1] >= PULSE_STEPS:
                    turned += swept
                    pulse = None
                else:
                    command, pulse = (-70, 70), (pulse[0], pulse[1] + 1)
        elif turned >= 2 * np.pi:
            sure = sure + 1 if est.confidence >= HOME_TRUST else 0
            if sure >= 20:
                return est
        sensors.send(*command, est.look_around or 0.0)
        due += DT
        r.wait(max(0.0, due - time.monotonic()))
    return None

