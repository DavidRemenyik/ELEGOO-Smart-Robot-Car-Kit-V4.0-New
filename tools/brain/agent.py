"""Talking, seeing robot: LiveKit Agents driving the car with an STT -> LLM -> TTS pipeline.

Run on the Mac (mic + speakers):  .venv/bin/python agent.py console
Type instead of talking:          .venv/bin/python agent.py console --text
Control pane (see + drive):       http://localhost:8765 while the agent runs
Without the car:                  .venv/bin/python fake_car.py --port 18100 --camera-port 18101
                                  CAR_HOST=127.0.0.1 CAR_PORT=18100 \
                                  CAR_CAMERA_URL=http://127.0.0.1:18101/capture \
                                  CAR_STREAM_URL=http://127.0.0.1:18101/stream .venv/bin/python agent.py console --text

Pipeline: Google Cloud Speech-to-Text hears the mic, Gemini on Vertex AI (project from
GOOGLE_CLOUD_PROJECT, default agrointel-dd8f6, via gcloud application-default credentials)
reads the words plus a fresh car camera picture on every turn and picks the moves, and
ElevenLabs speaks the reply (CAR_VOICE=google switches to Google Cloud TTS).
The ElevenLabs key comes from ELEVENLABS_API_KEY / ELEVEN_API_KEY, else from Secret Manager
(gcloud secrets versions access latest --secret=ELEVENLABS_API_KEY); it is kept in memory only.
Other settings: GEMINI_MODEL, GEMINI_LOCATION, ELEVEN_VOICE_ID, ELEVEN_MODEL.

drive.py steers each move from the Mac with the gyro and range sensor. The firmware stops the
wheels when commands stop arriving, and forward moves stop when the ultrasonic sensor sees
something closer than STOP_CM.
"""

import asyncio
import base64
import logging
import os
import subprocess
import threading
import time
import urllib.request
from typing import Literal

from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, ToolError, function_tool
from livekit.agents.llm import ImageContent
from livekit.plugins import elevenlabs, google, silero

import behaviors
from drive import Driver
from dashboard import Dashboard, Pilot, Recorder, Telemetry
from robot import DEFAULT_HOST, Robot

HOST = os.environ.get("CAR_HOST", DEFAULT_HOST)
PORT = int(os.environ.get("CAR_PORT", "100"))
CAMERA_URL = os.environ.get("CAR_CAMERA_URL", f"http://{HOST}/capture")
STREAM_URL = os.environ.get("CAR_STREAM_URL", f"http://{HOST}:81/stream")
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "8765"))
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "agrointel-dd8f6")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
LOCATION = os.environ.get("GEMINI_LOCATION", "global")  # gemini-3.8-flash is not served in us-central1 for this project
CAR_VOICE = os.environ.get("CAR_VOICE", "elevenlabs")
ELEVEN_VOICE_ID = os.environ.get("ELEVEN_VOICE_ID", "cgSgspJ2msm6clMCkdW9")  # premade "Jessica - Playful, Bright, Warm"
ELEVEN_MODEL = os.environ.get("ELEVEN_MODEL", "eleven_flash_v2_5")  # ElevenLabs' ultra-low-latency model
GOOGLE_VOICE = os.environ.get("GOOGLE_TTS_VOICE", "en-US-Chirp3-HD-Puck")
CAMERA_FPS = 1.0  # refresh rate of the control pane's still frame

STOP_CM = 25
SETTLE_S = 1.2  # let a fresh camera frame reach the model after moving
REPEAT_S = 3.0  # Gemini Live sometimes re-issues the move it just made; an identical move this soon is dropped once
MANUAL_SPEED = 150
HUMAN_HOLD_S = 1.5  # after the human touches the controls, the agent keeps its wheels still this long
RECONNECT_S = 3.0

log = logging.getLogger("car-agent")

INSTRUCTIONS = """You are Elegoo, a small, friendly four-wheeled robot car. You talk out loud in English,
in short, natural sentences, with no markdown, lists or emoji, because everything you write is spoken.
You see the world through a camera on your head: each time you think, you get a fresh picture of what
it sees right now. The camera sits low, about 15 cm above the floor, and points up at about 20 degrees,
so you see ceilings, walls and the upper parts of things, people usually look tall and you may only see
their legs, and the floor right in front of you is out of view: small things on the floor close by may
not show, so trust the distance sensor and move reports for nearby obstacles.
Your ears and voice are not on your body: the microphone and speaker are on a Mac computer. The
person you talk with sits near the Mac and may be in a different spot, or a different room, than
what your camera shows. Hearing someone clearly does not mean they are close to you.

How you move:
- Only move while you are doing a task someone gave you (find a person, find an object, come here,
  look around, patrol). With no task, stay still and just talk.
- drive_forward runs smoothly, holds its heading and slows down by itself before obstacles, so
  prefer one long move (up to 300 cm) over many short ones. turn reports the angle really turned.
  After each move, look at the new camera view before deciding the next move.
- explore drives around on its own for a while; use it to patrol or to cover ground while
  searching. It keeps running after the tool returns: watch the camera, and call stop to end it.
  Any other move or look also ends it. You are told when it ends.
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


def _eleven_key():
    key = os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("ELEVEN_API_KEY")
    if key:
        return key
    cmd = ["gcloud", "secrets", "versions", "access", "latest", "--secret=ELEVENLABS_API_KEY", f"--project={PROJECT}"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Cannot read the ElevenLabs key from Secret Manager: {e.stderr.strip()}") from e
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"Cannot run gcloud to read the ElevenLabs key: {e}") from e
    return out.stdout.strip()


def make_tts():
    if CAR_VOICE == "google":
        return google.TTS(voice_name=GOOGLE_VOICE, language="en-US")
    if CAR_VOICE != "elevenlabs":
        raise SystemExit(f"CAR_VOICE must be elevenlabs or google, not {CAR_VOICE!r}")
    return elevenlabs.TTS(voice_id=ELEVEN_VOICE_ID, model=ELEVEN_MODEL, api_key=_eleven_key())


def make_llm():
    return google.LLM(
        vertexai=True,
        project=PROJECT,
        location=LOCATION,
        model=MODEL,
        thinking_config={"thinking_level": "low"},  # less thinking, sooner first word; "minimal" is refused by this model
    )


class CarCamera:
    """Fetches a fresh camera still for every model turn and keeps the control pane's frame current."""

    def __init__(self, url, telemetry):
        self._url = url
        self._telemetry = telemetry
        self._failing = False
        self._task = None

    async def snapshot(self):
        try:
            jpeg = await asyncio.to_thread(_fetch, self._url)
        except OSError as e:
            if not self._failing:
                log.warning("camera %s unreachable: %s", self._url, e)
            self._failing = True
            return None
        if self._failing:
            log.info("camera back")
            self._failing = False
        self._telemetry.set_frame(jpeg)
        return jpeg

    def start(self):
        self._task = asyncio.create_task(self._refresh())

    async def _refresh(self):
        while True:
            await self.snapshot()
            await asyncio.sleep(1.0 / CAMERA_FPS)

    def close(self):
        if self._task:
            self._task.cancel()


class Car:
    def __init__(self, host, port, telemetry):
        self.host = host
        self.port = port
        self.telemetry = telemetry
        self._robot = None
        self._lock = threading.Lock()
        self._retry_at = 0.0
        self._human_at = 0.0
        self.cancel = threading.Event()

    def human_input(self):
        self._human_at = time.monotonic()

    def human_active(self):
        return time.monotonic() - self._human_at < HUMAN_HOLD_S

    def should_stop(self):
        if self.cancel.is_set():
            return "I was told to stop"
        if self.human_active():
            return "the human took over the controls"
        return None

    def _call(self, fn, by):
        try:
            if self._robot is None:
                self._robot = Robot.wifi(self.host, self.port)
            return fn(Recorder(self._robot, self.telemetry, by))
        except (OSError, TimeoutError) as e:
            self._drop()
            self._retry_at = time.monotonic() + RECONNECT_S
            self.telemetry.link.update(state="offline", error=str(e))
            self.telemetry.touch()
            raise

    def run(self, fn, by="agent"):
        with self._lock:
            try:
                return self._call(fn, by)
            except (OSError, TimeoutError) as e:
                raise ToolError(f"I can't reach my wheels right now ({e}).") from e

    def run_quiet(self, fn, by="human"):
        with self._lock:
            return self._quiet(fn, by)

    def try_run(self, fn):
        if not self._lock.acquire(blocking=False):
            return None
        try:
            return self._quiet(fn, None)
        finally:
            self._lock.release()

    def _quiet(self, fn, by):
        if self._robot is None and time.monotonic() < self._retry_at:
            return None
        try:
            return self._call(fn, by)
        except (OSError, TimeoutError):
            return None

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


def _forward(robot, cm, should_stop):
    gone, why = Driver(robot, should_stop).forward(cm, STOP_CM)
    if why:
        return f"Stopped after about {gone:.0f} cm: {why}."
    return f"Drove forward about {gone:.0f} cm. Clear distance ahead is {robot.distance()} cm."


def _back(robot, cm, should_stop):
    gone, why = Driver(robot, should_stop).back(cm)
    return f"Backed up about {gone:.0f} cm" + (f", then stopped: {why}." if why else ".")


def _turn(robot, degrees, should_stop):
    turned, why = Driver(robot, should_stop).turn(degrees)
    side = "left" if turned > 0 else "right"
    return f"Turned {abs(turned):.0f} degrees {side}" + (f", then stopped: {why}." if why else ".")


def _explore(robot, seconds, should_stop):
    driven, turns, why = Driver(robot, should_stop).explore(seconds, STOP_CM)
    return f"Explored: drove about {driven:.0f} cm and changed direction {turns} times. " + (f"Ended early: {why}." if why else "Time is up.")


def _look(robot, degrees):
    robot.head(behaviors.CENTER + degrees)
    return f"Head turned {abs(degrees)} degrees {'left' if degrees > 0 else 'right'}." if degrees else "Head facing forward."


def _scan(robot):
    readings = behaviors.scan(robot)
    parts = [f"{angle - behaviors.CENTER:+d} deg: {cm} cm" for angle, cm in readings]
    return "Free distance by head angle (+ is left): " + ", ".join(parts)


class CarAgent(Agent):
    def __init__(self, car, camera):
        super().__init__(instructions=INSTRUCTIONS)
        self.car = car
        self.camera = camera
        self.telemetry = car.telemetry
        self._last_move = None
        self._exploring = None

    async def llm_node(self, chat_ctx, tools, model_settings):
        # Every model call, including the one right after a move, sees the camera as it is now.
        # The picture goes into a copy of the context, so old frames never pile up in the history.
        jpeg = await self.camera.snapshot()
        chat_ctx = chat_ctx.copy()
        if jpeg:
            image = ImageContent(image="data:image/jpeg;base64," + base64.b64encode(jpeg).decode())
            chat_ctx.add_message(role="user", content=["[Your camera view right now]", image])
        else:
            chat_ctx.add_message(role="user", content=["[Your camera is not answering right now; you cannot see.]"])
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk

    async def _end_exploring(self):
        if self._exploring and not self._exploring.done():
            self.car.cancel.set()
            await self._exploring

    async def _explore(self, seconds):
        try:
            result = await self._act(f"Explore {seconds} s", lambda r: _explore(r, seconds, self.car.should_stop), settle=False)
        except ToolError as e:
            result = str(e)
        if not self.car.cancel.is_set():
            self.session.generate_reply(instructions=f"Your exploring ended. {result} Tell the person briefly what you saw on the way.")

    async def _drive(self, label, move, fn):
        if not self.telemetry.agent["autonomy"]:
            return self._refuse(label, "Not moved: the human has paused my driving from the control pane.")
        if self.car.human_active():
            return self._refuse(label, "Not moved: the human is driving me right now. Wait and watch the camera.")
        last, self._last_move = self._last_move, None
        if last and last[0] == move and time.monotonic() - last[1] < REPEAT_S:
            return self._refuse(label, "Skipped: that exact move just finished. Look at the camera view; if you still want it, call it again.")
        await self._end_exploring()
        self.car.cancel.clear()
        if fn is None:
            self._exploring = asyncio.create_task(self._explore(move[1]))
            self._last_move = (move, time.monotonic())
            return "Exploring on my own now. Watch the camera; call stop to end it early."
        result = await self._act(label, fn)
        self._last_move = (move, time.monotonic())
        return result

    def _refuse(self, label, reason):
        self.telemetry.tool_end(self.telemetry.tool_start(label), reason, ok=False)
        return reason

    async def _act(self, label, fn, settle=True):
        entry = self.telemetry.tool_start(label)
        try:
            result = await asyncio.to_thread(self.car.run, fn)
        except ToolError as e:
            self.telemetry.tool_end(entry, str(e), ok=False)
            raise
        self.telemetry.tool_end(entry, result)
        if settle:
            await asyncio.sleep(SETTLE_S)
        return result

    @function_tool()
    async def drive_forward(self, centimeters: int) -> str:
        """Drive straight forward, smoothly. Slows down and stops early if something is in the way.

        Args:
            centimeters: distance to drive, 5 to 300.
        """
        cm = max(5, min(300, centimeters))
        return await self._drive(f"Drive forward {cm} cm", ("forward", cm), lambda r: _forward(r, cm, self.car.should_stop))

    @function_tool()
    async def drive_back(self, centimeters: int) -> str:
        """Back up straight. There is no sensor behind, so keep it short.

        Args:
            centimeters: distance to back up, 5 to 40.
        """
        cm = max(5, min(40, centimeters))
        return await self._drive(f"Back up {cm} cm", ("back", cm), lambda r: _back(r, cm, self.car.should_stop))

    @function_tool()
    async def turn(self, direction: Literal["left", "right"], degrees: int) -> str:
        """Spin the whole car in place.

        Args:
            direction: which way to spin, from the car's point of view.
            degrees: how far to spin, 5 to 180.
        """
        deg = max(5, min(180, abs(degrees))) * (1 if direction == "left" else -1)
        return await self._drive(f"Turn {direction} {abs(deg)}°", ("turn", deg), lambda r: _turn(r, deg, self.car.should_stop))

    @function_tool()
    async def explore(self, seconds: int) -> str:
        """Drive around on your own, avoiding obstacles, to patrol or to cover ground while searching.

        Returns at once while the car keeps driving. You are told when it ends.

        Args:
            seconds: how long to explore, 10 to 120.
        """
        secs = max(10, min(120, seconds))
        return await self._drive(f"Explore {secs} s", ("explore", secs), None)

    @function_tool()
    async def look_direction(self, direction: Literal["left", "right", "forward"], degrees: int = 45) -> str:
        """Pan only the head (camera and distance sensor); the car stays still.

        Args:
            direction: where to point the head, from the car's point of view.
            degrees: how far from straight ahead, 10 to 80. Ignored for forward.
        """
        deg = 0 if direction == "forward" else max(10, min(80, abs(degrees))) * (1 if direction == "left" else -1)
        label = "Look forward" if deg == 0 else f"Look {direction} {abs(deg)}°"
        await self._end_exploring()
        return await self._act(label, lambda r: _look(r, deg))

    @function_tool()
    async def scan_distances(self) -> str:
        """Sweep the head left to right and measure free space in each direction, in cm."""
        await self._end_exploring()
        return await self._act("Scan distances", _scan, settle=False)

    @function_tool()
    async def stop(self) -> str:
        """Stop all motion immediately."""
        self.car.cancel.set()
        await self._end_exploring()
        await self._act("Stop", lambda r: r.stop(), settle=False)
        return "Stopped."


server = AgentServer()


class PaneActions:
    def __init__(self, session, telemetry):
        self.session = session
        self.telemetry = telemetry

    async def estop(self):
        self.telemetry.agent["autonomy"] = False
        await self.session.interrupt(force=True)

    def set_mic(self, on):
        self.session.input.set_audio_enabled(on)
        self.telemetry.agent["mic"] = on
        self.telemetry.touch()

    def say(self, text):
        self.session.generate_reply(user_input=text)


def _follow_session(session, telemetry):
    def agent_state(ev):
        telemetry.agent["state"] = ev.new_state
        telemetry.touch()

    def user_state(ev):
        telemetry.agent["user_state"] = ev.new_state
        telemetry.touch()

    def heard(ev):
        if ev.transcript.strip():
            telemetry.say("user", ev.transcript, ev.is_final)

    def item(ev):
        role, text = getattr(ev.item, "role", None), getattr(ev.item, "text_content", None)
        if role in ("user", "assistant") and text:
            telemetry.say("user" if role == "user" else "agent", text)

    session.on("agent_state_changed", agent_state)
    session.on("user_state_changed", user_state)
    session.on("user_input_transcribed", heard)
    session.on("conversation_item_added", item)


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    telemetry = Telemetry(STOP_CM)
    car = Car(HOST, PORT, telemetry)
    ctx.add_shutdown_callback(lambda: asyncio.to_thread(car.close))
    camera = CarCamera(CAMERA_URL, telemetry)
    session = AgentSession(
        stt=google.STT(languages="en-US", model="latest_long"),
        llm=make_llm(),
        tts=make_tts(),
        vad=silero.VAD.load(),
    )
    _follow_session(session, telemetry)
    await session.start(room=ctx.room, agent=CarAgent(car, camera))
    camera.start()
    ctx.add_shutdown_callback(camera.close)
    pane = Dashboard(telemetry, Pilot(car, telemetry, MANUAL_SPEED, behaviors.scan), PaneActions(session, telemetry), STREAM_URL, DASHBOARD_PORT)
    await pane.start()
    ctx.add_shutdown_callback(pane.close)
    await session.generate_reply(instructions="Say a short hello and that you can look around, find people, and find things.")


if __name__ == "__main__":
    agents.cli.run_app(server)
