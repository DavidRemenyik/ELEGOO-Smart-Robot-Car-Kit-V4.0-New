"""Talking, seeing robot: LiveKit Agents + Gemini Live (Vertex AI) driving the car.

Run on the Mac (mic + speakers):  .venv/bin/python agent.py console
Type instead of talking:          .venv/bin/python agent.py console --text

Gemini Live hears the mic, watches the car camera (about one frame per second)
and drives through the tools below. Every move is a short timed command that
the car's firmware stops by itself, and forward moves stop when the ultrasonic
sensor sees something closer than STOP_CM.
"""

import asyncio
import io
import logging
import os
import threading
import time
import urllib.request
from typing import Literal

from livekit import agents, rtc
from livekit.agents import Agent, AgentServer, AgentSession, ToolError, function_tool
from livekit.agents.voice.io import VideoInput
from livekit.plugins import google
from PIL import Image

import behaviors
from robot import DEFAULT_HOST, Robot

HOST = os.environ.get("CAR_HOST", DEFAULT_HOST)
PORT = int(os.environ.get("CAR_PORT", "100"))
CAMERA_URL = os.environ.get("CAR_CAMERA_URL", f"http://{HOST}/capture")
MODEL = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.8-live")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")  # Live is not served on "global"
VOICE = os.environ.get("GEMINI_VOICE", "Puck")

SPEED = 110
CM_PER_S = 30.0  # forward speed at SPEED; calibrate on the floor
DEG_PER_S = 120.0  # spin speed at SPEED; calibrate on the floor
STEP_MS = 200  # forward moves run in steps this long, checking range and floor sensors between steps
STOP_CM = 25
SETTLE_S = 1.2  # let a fresh camera frame reach the model after moving
REPEAT_S = 3.0  # Gemini Live sometimes re-issues the move it just made; an identical move this soon is dropped once

log = logging.getLogger("car-agent")

INSTRUCTIONS = """You are Elegoo, a small, friendly four-wheeled robot car. You talk out loud in English,
in short, natural sentences. You see the world live through a camera on your head, mounted low,
about 15 cm above the floor, so people usually look tall and you may only see their legs.

How you move:
- Only move while you are doing a task someone gave you (find a person, find an object, come here,
  look around). With no task, stay still and just talk.
- Use drive_forward / drive_back with small distances (10 to 60 cm) and turn for rotations.
  After each move, look at the new camera view before deciding the next move.
- look_direction pans only your head and camera, which is quicker than turning the whole car.
- If a move reports it was blocked, do not repeat it; turn and find another way.
- When someone says stop, call stop immediately.

Finding things:
- To search, look left and right with look_direction, then turn about 60 degrees at a time and check
  the view after each step. A full circle is about 6 steps. Say briefly what you see as you go.
- When you spot the target, turn so it is in the middle of the view and drive toward it in short
  steps, stopping about 30 to 50 cm away.
- When you find a person, greet them warmly, introduce yourself, and chat. When you find an object,
  say where it is.
- If you cannot find it after a full circle and a couple of moves, say so and ask for a hint.

Never claim to see something you are not seeing. If the camera view is dark or blurry, say so.
"""


def _fetch(url, timeout=3.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


class CarCamera(VideoInput):
    def __init__(self, url, fps=1.0):
        super().__init__(label="car-camera")
        self._url = url
        self._period = 1.0 / fps
        self._next = 0.0
        self._failing = False

    async def __anext__(self) -> rtc.VideoFrame:
        while True:
            delay = self._next - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._next = time.monotonic() + self._period
            try:
                jpeg = await asyncio.to_thread(_fetch, self._url)
            except OSError as e:
                if not self._failing:
                    log.warning("camera %s unreachable: %s", self._url, e)
                self._failing = True
                continue
            if self._failing:
                log.info("camera back")
                self._failing = False
            img = Image.open(io.BytesIO(jpeg)).rotate(180).convert("RGBA")  # camera module is mounted upside down
            return rtc.VideoFrame(img.width, img.height, rtc.VideoBufferType.RGBA, img.tobytes())


class Car:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._robot = None
        self._lock = threading.Lock()

    def run(self, fn):
        with self._lock:
            try:
                if self._robot is None:
                    self._robot = Robot.wifi(self.host, self.port)
                return fn(self._robot)
            except (OSError, TimeoutError) as e:
                self._drop()
                raise ToolError(f"I can't reach my wheels right now ({e}).") from e

    def _drop(self):
        robot, self._robot = self._robot, None
        if robot is not None:
            try:
                robot.close()
            except (OSError, TimeoutError):
                pass

    def close(self):
        with self._lock:
            self._drop()


def _forward(robot, cm):
    robot.head(behaviors.CENTER)
    time.sleep(0.3)
    total_ms = int(cm / CM_PER_S * 1000)
    done_ms = 0
    while done_ms < total_ms:
        if not robot.on_ground():
            return f"Stopped after about {done_ms * CM_PER_S / 1000:.0f} cm: no floor under my front sensors (an edge, or I am lifted)."
        ahead = robot.distance()
        if ahead < STOP_CM:
            return f"Stopped after about {done_ms * CM_PER_S / 1000:.0f} cm: obstacle {ahead} cm ahead."
        step = min(STEP_MS, total_ms - done_ms)
        robot.move("forward", SPEED, step)  # returns when the car has stopped itself
        done_ms += step
    return f"Drove forward about {cm} cm. Clear distance ahead was {robot.distance()} cm."


def _back(robot, cm):
    robot.move("back", SPEED, int(cm / CM_PER_S * 1000))
    return f"Backed up about {cm} cm."


def _turn(robot, degrees):
    robot.move("left" if degrees > 0 else "right", SPEED, int(abs(degrees) / DEG_PER_S * 1000))
    return f"Turned about {abs(degrees)} degrees {'left' if degrees > 0 else 'right'}."


def _look(robot, degrees):
    robot.head(behaviors.CENTER + degrees)
    return f"Head turned {abs(degrees)} degrees {'left' if degrees > 0 else 'right'}." if degrees else "Head facing forward."


def _scan(robot):
    readings = behaviors.scan(robot)
    parts = [f"{angle - behaviors.CENTER:+d} deg: {cm} cm" for angle, cm in readings]
    return "Free distance by head angle (+ is left): " + ", ".join(parts)


class CarAgent(Agent):
    def __init__(self, car):
        super().__init__(instructions=INSTRUCTIONS)
        self.car = car
        self._last_move = None

    async def _drive(self, move, fn):
        last, self._last_move = self._last_move, None
        if last and last[0] == move and time.monotonic() - last[1] < REPEAT_S:
            return "Skipped: that exact move just finished. Look at the camera view; if you still want it, call it again."
        result = await self._act(fn)
        self._last_move = (move, time.monotonic())
        return result

    async def _act(self, fn, settle=True):
        result = await asyncio.to_thread(self.car.run, fn)
        if settle:
            await asyncio.sleep(SETTLE_S)
        return result

    @function_tool()
    async def drive_forward(self, centimeters: int) -> str:
        """Drive straight forward. Stops early if something is in the way.

        Args:
            centimeters: distance to drive, 5 to 100.
        """
        cm = max(5, min(100, centimeters))
        return await self._drive(("forward", cm), lambda r: _forward(r, cm))

    @function_tool()
    async def drive_back(self, centimeters: int) -> str:
        """Back up straight. There is no sensor behind, so keep it short.

        Args:
            centimeters: distance to back up, 5 to 40.
        """
        cm = max(5, min(40, centimeters))
        return await self._drive(("back", cm), lambda r: _back(r, cm))

    @function_tool()
    async def turn(self, direction: Literal["left", "right"], degrees: int) -> str:
        """Spin the whole car in place.

        Args:
            direction: which way to spin, from the car's point of view.
            degrees: how far to spin, 5 to 180.
        """
        deg = max(5, min(180, abs(degrees))) * (1 if direction == "left" else -1)
        return await self._drive(("turn", deg), lambda r: _turn(r, deg))

    @function_tool()
    async def look_direction(self, direction: Literal["left", "right", "forward"], degrees: int = 45) -> str:
        """Pan only the head (camera and distance sensor); the car stays still.

        Args:
            direction: where to point the head, from the car's point of view.
            degrees: how far from straight ahead, 10 to 80. Ignored for forward.
        """
        deg = 0 if direction == "forward" else max(10, min(80, abs(degrees))) * (1 if direction == "left" else -1)
        return await self._act(lambda r: _look(r, deg))

    @function_tool()
    async def scan_distances(self) -> str:
        """Sweep the head left to right and measure free space in each direction, in cm."""
        return await self._act(_scan, settle=False)

    @function_tool()
    async def stop(self) -> str:
        """Stop all motion immediately."""
        await self._act(lambda r: r.stop(), settle=False)
        return "Stopped."


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    car = Car(HOST, PORT)
    ctx.add_shutdown_callback(lambda: asyncio.to_thread(car.close))
    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            vertexai=True,
            location=LOCATION,
            model=MODEL,
            voice=VOICE,
            language="en-US",
        ),
    )
    await session.start(room=ctx.room, agent=CarAgent(car))
    session.input.video = CarCamera(CAMERA_URL)
    await session.generate_reply(instructions="Say a short hello and that you can look around, find people, and find things.")


if __name__ == "__main__":
    agents.cli.run_app(server)
