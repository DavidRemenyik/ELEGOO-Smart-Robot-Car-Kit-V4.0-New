"""Talking, seeing robot: LiveKit Agents driving the car with Gemini Live (or an STT -> LLM -> TTS pipeline).

Run on the Mac (mic + speakers):  .venv/bin/python agent.py console
Type instead of talking:          .venv/bin/python agent.py console --text
Control pane (see + drive):       http://localhost:8765 while the agent runs
Without the car:                  .venv/bin/python fake_car.py --port 18100 --camera-port 18101
                                  CAR_HOST=127.0.0.1 CAR_PORT=18100 \
                                  CAR_CAMERA_URL=http://127.0.0.1:18101/capture \
                                  CAR_STREAM_URL=http://127.0.0.1:18101/stream .venv/bin/python agent.py console --text

Google Cloud: Vertex AI in project GOOGLE_CLOUD_PROJECT (default agrointel-dd8f6) via gcloud
application-default credentials.

CAR_VOICE=gemini-live (default): one realtime model (GEMINI_LIVE_MODEL, default gemini-3.8-live,
in GEMINI_LIVE_LOCATION, default us-central1) hears the mic, watches the car camera (one frame a
second, plus a fresh frame right after every move) and speaks with a Google voice (GEMINI_VOICE,
default Zephyr). Fastest replies.

CAR_VOICE=elevenlabs: ElevenLabs speech-to-text hears the mic, Gemini (GEMINI_MODEL in
GEMINI_LOCATION) reads the words plus a fresh car camera picture on every turn and picks the moves,
and ElevenLabs speaks the reply (ELEVEN_VOICE_ID, ELEVEN_MODEL). The ElevenLabs key comes from
ELEVENLABS_API_KEY / ELEVEN_API_KEY, else from Secret Manager
(gcloud secrets versions access latest --secret=ELEVENLABS_API_KEY); it is kept in memory only.

find_object always asks GEMINI_MODEL in GEMINI_LOCATION where the object is.

Following a person and patrol sightings need the person detector (3.7 MB, once per venv):
  mkdir -p .venv/share/elegoo && curl -fsSL -o .venv/share/elegoo/yolox_nano.onnx \
    https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx

drive.py steers each move from the Mac with the gyro and range sensor. The firmware stops the
wheels when commands stop arriving, and forward moves stop when the ultrasonic sensor sees
something closer than STOP_CM.
"""

import asyncio
import base64
import functools
import io
import logging
import os
import subprocess
import threading
import time
from typing import Literal

from livekit import agents, rtc
from livekit.agents import Agent, AgentServer, AgentSession, ToolError, function_tool
from livekit.agents.llm import ImageContent
from livekit.agents.voice.io import VideoInput
from google.genai import types as genai
from livekit.plugins import elevenlabs, google, silero
from PIL import Image

import behaviors
from drive import Driver
from dashboard import Dashboard, Pilot, Recorder, Telemetry
from robot import DEFAULT_HOST, Robot
from tasks import ObjectModel, TaskManager, find_object, follow_person, patrol
from vision import Vision

HOST = os.environ.get("CAR_HOST", DEFAULT_HOST)
PORT = int(os.environ.get("CAR_PORT", "100"))
CAMERA_URL = os.environ.get("CAR_CAMERA_URL", f"http://{HOST}/capture")
STREAM_URL = os.environ.get("CAR_STREAM_URL", f"http://{HOST}:81/stream")
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "8765"))
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "agrointel-dd8f6")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
LOCATION = os.environ.get("GEMINI_LOCATION", "global")  # gemini-3.8-flash is not served in us-central1 for this project
CAR_VOICE = os.environ.get("CAR_VOICE", "gemini-live")
LIVE_MODEL = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.8-live")
LIVE_LOCATION = os.environ.get("GEMINI_LIVE_LOCATION", "us-central1")  # Live is not served on "global"
LIVE_VOICE = os.environ.get("GEMINI_VOICE", "Zephyr")  # Google's "bright" voice, closest to the ElevenLabs one
LIVE_FPS = 1.0  # Gemini Live takes about one picture a second
ELEVEN_VOICE_ID = os.environ.get("ELEVEN_VOICE_ID", "cgSgspJ2msm6clMCkdW9")  # premade "Jessica - Playful, Bright, Warm"
ELEVEN_MODEL = os.environ.get("ELEVEN_MODEL", "eleven_flash_v2_5")  # ElevenLabs' ultra-low-latency model
CAMERA_FPS = 5.0  # refresh rate of the control pane's still frame
SNAPSHOT_MAX_AGE_S = 3.0  # the model never gets a picture older than this; the real camera needs up to 2 s per picture
SNAPSHOT_WAIT_S = 6.0  # how long a model turn waits for the camera before going on blind

STOP_CM = 25
ENDPOINT_MIN_S = 0.9  # silence before a turn counts as finished; shorter cut the speaker off mid-sentence
SETTLE_S = 0.4  # let the car and the head come to rest before the model's next picture
MANUAL_SPEED = 150
HUMAN_HOLD_S = 1.5  # after the human touches the controls, the agent keeps its wheels still this long
RECONNECT_S = 3.0
REPEAT_S = 3.0  # Gemini Live re-issues the move it just made; an identical move this soon is dropped once
INITIATIVE = os.environ.get("CAR_INITIATIVE", "0") == "1"

log = logging.getLogger("car-agent")

INSTRUCTIONS = """You are Elegoo, a small, friendly four-wheeled robot car. You talk out loud in English,
in short, natural sentences, with no markdown, lists or emoji, because everything you write is spoken.
You see the world through a camera on your head, and after every move or look you get a fresh picture
of what it sees then. The camera sits low, about 15 cm above the floor, and points up at about 20 degrees,
so you see ceilings, walls and the upper parts of things, people usually look tall and you may only see
their legs, and the floor right in front of you is out of view: small things on the floor close by may
not show, so trust the distance sensor and move reports for nearby obstacles.
Your ears and voice are not on your body: the microphone and speaker are on a Mac computer. The
person you talk with sits near the Mac and may be in a different spot, or a different room, than
what your camera shows. Hearing someone clearly does not mean they are close to you.

How you move:
- Only move on a request. An independent, owner-enabled schedule handles self-started actions;
  never start one yourself through a tool. Do not offer navigation to named places.
- drive_forward runs smoothly, holds its heading and slows down by itself before obstacles, so
  prefer one long move (up to 300 cm) over many short ones. turn reports the angle really turned.
  After each move, look at the new camera view before deciding the next move.
- Use follow_person, patrol, or find_object directly for those jobs, not chains of moves, looks
  or explore. They run in the background and speak their own factual results. Confirm the start
  briefly, then let them work. Use task_status to check without interrupting. A new motion tool
  replaces the current job, so don't call moves or looks merely to check its progress.
- explore is for unstructured roaming, not patrol or search.
- look_direction pans only your head and camera, which is quicker than turning the whole car.
- If a move reports it was blocked, do not repeat it; turn and find another way.
- When someone says stop, call stop immediately.

Never claim to see something you are not seeing. If the camera view is dark or blurry, say so.
"""


@functools.cache
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
    return elevenlabs.TTS(voice_id=ELEVEN_VOICE_ID, model=ELEVEN_MODEL, api_key=_eleven_key())


def make_llm():
    return google.LLM(
        vertexai=True,
        project=PROJECT,
        location=LOCATION,
        model=MODEL,
        thinking_config={"thinking_level": "low"},  # less thinking, sooner first word; "minimal" is refused by this model
    )


def make_session():
    if CAR_VOICE == "gemini-live":
        return AgentSession(
            llm=google.realtime.RealtimeModel(
                vertexai=True,
                project=PROJECT,
                location=LIVE_LOCATION,
                model=LIVE_MODEL,
                voice=LIVE_VOICE,
                language="en-US",
                # 3.8 Live defaults to NON_BLOCKING tools and then re-issued the same move up to 6 times
                tool_behavior=genai.Behavior.BLOCKING,
                realtime_input_config=genai.RealtimeInputConfig(
                    automatic_activity_detection=genai.AutomaticActivityDetection(silence_duration_ms=int(ENDPOINT_MIN_S * 1000)),
                ),
                # Without compression Vertex ends an audio+video session after about 2 minutes; a small window keeps replies quick
                context_window_compression=genai.ContextWindowCompressionConfig(
                    trigger_tokens=32000, sliding_window=genai.SlidingWindow(target_tokens=16000),
                ),
                session_resumption=genai.SessionResumptionConfig(),  # reconnects every ~10 min keep the conversation
            ),
            video_sampler=None,  # LiveVideo paces the frames itself and must not lose the one after a move
        )
    if CAR_VOICE == "elevenlabs":
        return AgentSession(
            stt=elevenlabs.STT(api_key=_eleven_key(), language_code="en", use_realtime=True),
            min_endpointing_delay=ENDPOINT_MIN_S,
            llm=make_llm(),
            tts=make_tts(),
            vad=silero.VAD.load(),
        )
    raise SystemExit(f"CAR_VOICE must be gemini-live or elevenlabs, not {CAR_VOICE!r}")


class CarCamera:
    """Hands the newest camera still to the model and the control pane.

    The car's camera serves one request at a time, so Vision is its only reader and everything else
    takes frames from it.
    """

    def __init__(self, url, stream_url, telemetry):
        self.vision = Vision(url, stream_url)
        self._telemetry = telemetry
        self._failing = False
        self._task = None
        self._moved_at = 0.0
        self.live = False  # a LiveVideo streams frames to the realtime model
        self._live_kick = asyncio.Event()
        self._live_due = 0.0
        self._live_sent = 0.0  # capture time of the newest frame handed to the realtime model

    def moved(self):
        """The view just changed: only pictures requested from now on may be shown to the model."""
        self._moved_at = time.monotonic()

    async def after_move(self):
        """With live video, hand the model a picture taken after the move before it hears the move's result."""
        if not self.live:
            return
        self._live_kick.set()
        deadline = time.monotonic() + SNAPSHOT_WAIT_S
        while self._live_sent < self._moved_at and time.monotonic() < deadline:
            await asyncio.sleep(0.05)

    async def live_frame(self):
        """Next picture for the realtime model: one a second, or at once after a move."""
        try:
            await asyncio.wait_for(self._live_kick.wait(), max(0.0, self._live_due - time.monotonic()))
        except TimeoutError:
            pass
        self._live_kick.clear()
        self._live_due = time.monotonic() + 1.0 / LIVE_FPS
        frame = await self._fresh()
        if frame:
            self._live_sent = frame.timestamp
        return frame

    async def snapshot(self):
        frame = await self._fresh()
        return frame.jpeg if frame else None

    async def _fresh(self):
        self.vision.start()
        deadline = time.monotonic() + SNAPSHOT_WAIT_S
        while not ((frame := self.vision.newest(SNAPSHOT_MAX_AGE_S)) and frame.timestamp >= self._moved_at):
            if time.monotonic() > deadline:
                if not self._failing:
                    log.warning("no camera frame for %.0f s", SNAPSHOT_WAIT_S)
                self._failing = True
                return None
            await asyncio.sleep(0.05)
        if self._failing:
            log.info("camera back")
            self._failing = False
        self._telemetry.set_frame(frame.jpeg)
        return frame

    def start(self):
        self._task = asyncio.create_task(self._refresh())

    async def _refresh(self):
        while True:
            await self.snapshot()
            await asyncio.sleep(1.0 / CAMERA_FPS)

    def close(self):
        if self._task:
            self._task.cancel()


def _rgba(jpeg):
    with Image.open(io.BytesIO(jpeg)) as image:
        rgba = image.convert("RGBA")
    return rtc.VideoFrame(rgba.width, rgba.height, rtc.VideoBufferType.RGBA, rgba.tobytes())


class LiveVideo(VideoInput):
    """The realtime model's video input, fed from the camera's single reader."""

    def __init__(self, camera):
        super().__init__(label="car-camera")
        self.camera = camera
        camera.live = True

    async def __anext__(self) -> rtc.VideoFrame:
        while not (frame := await self.camera.live_frame()):
            pass  # live_frame already waited for the camera and logged the outage
        return await asyncio.to_thread(_rgba, frame.jpeg)


class Car:
    def __init__(self, host, port, telemetry):
        self.host = host
        self.port = port
        self.telemetry = telemetry
        self._robot = None
        self._lock = threading.Lock()
        self._retry_at = 0.0
        self._human_at = 0.0
        self.human_epoch = 0

    def human_input(self):
        self._human_at = time.monotonic()
        self.human_epoch += 1

    def human_active(self):
        return time.monotonic() - self._human_at < HUMAN_HOLD_S

    def _call(self, fn, by):
        try:
            if self._robot is None:
                self._robot = Robot.wifi(self.host, self.port)
            return fn(Recorder(self._robot, self.telemetry, by))
        except (OSError, TimeoutError) as e:
            self._drop()
            self._retry_at = time.monotonic() + RECONNECT_S
            self.telemetry.link.update(state="offline", error=str(e))
            self.telemetry.sensors["on_ground"] = None
            self.telemetry.moved(None, None)
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


def _forward(robot, cm):
    gone, why = Driver(robot, stall=robot.stall).forward(cm, STOP_CM)
    if why:
        return f"Stopped after about {gone:.0f} cm: {why}."
    return f"Drove forward about {gone:.0f} cm. Clear distance ahead is {robot.distance()} cm."


def _back(robot, cm):
    gone, why = Driver(robot, stall=robot.stall).back(cm)
    return f"Backed up about {gone:.0f} cm" + (f", then stopped: {why}." if why else ".")


def _turn(robot, degrees):
    turned, why = Driver(robot).turn(degrees)
    side = "left" if turned > 0 else "right"
    return f"Turned {abs(turned):.0f} degrees {side}" + (f", then stopped: {why}." if why else ".")


def _explore(robot, seconds):
    driven, turns, why = Driver(robot, stall=robot.stall).explore(seconds, STOP_CM)
    return f"Explored: drove about {driven:.0f} cm and changed direction {turns} times. " + (f"Ended early: {why}." if why else "Time is up.")


def _look(robot, degrees):
    robot.head(behaviors.CENTER + degrees)
    return f"Head turned {abs(degrees)} degrees {'left' if degrees > 0 else 'right'}." if degrees else "Head facing forward."


def _scan(robot):
    readings = []
    for angle in range(10, 171, 20):
        robot.head(angle)
        robot.wait(0.3)
        readings.append((angle, robot.distance()))
    robot.head(behaviors.CENTER)
    parts = [f"{angle - behaviors.CENTER:+d} deg: {cm} cm" for angle, cm in readings]
    return "Free distance by head angle (+ is left): " + ", ".join(parts)


class CarAgent(Agent):
    def __init__(self, car, camera):
        super().__init__(instructions=INSTRUCTIONS)
        self.car = car
        self.camera = camera
        self.telemetry = car.telemetry
        self.jobs = TaskManager(car, camera.vision, self._speak_task)
        self.object_model = ObjectModel(PROJECT, LOCATION, MODEL)
        self._last_move = None  # (label, monotonic time it finished); cleared by every new user turn

    def new_user_turn(self):
        """The repeat guard only catches moves the model re-issues by itself, never ones the person asked for."""
        self._last_move = None

    async def on_enter(self):
        # On speech start, not the final transcript: Gemini Live often moves before the transcript is final.
        self.session.on("user_state_changed", lambda ev: ev.new_state == "speaking" and self.new_user_turn())

    async def _speak_task(self, text: str) -> bool:
        if self.session.tts:
            speech = self.session.say(text, allow_interruptions=True)
        else:  # Gemini Live has no separate voice to read text aloud, so the model says it
            speech = self.session.generate_reply(
                instructions=f"Say exactly this to the person, word for word, and call no tools: {text}",
                allow_interruptions=True,
            )
        try:
            await speech.wait_for_playout()
        except asyncio.CancelledError:
            speech.interrupt()
            raise
        return not speech.interrupted

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

    async def _drive(self, label, fn):
        if not self.telemetry.agent["autonomy"]:
            return self._refuse(label, "Not moved: the human has paused my driving from the control pane.")
        if self.car.human_active():
            return self._refuse(label, "Not moved: the human is driving me right now. Wait and watch the camera.")
        last, self._last_move = self._last_move, None
        if last and last[0] == label and time.monotonic() - last[1] < REPEAT_S:
            return self._refuse(label, "Skipped: that exact move just finished. Look at the camera view; if you still want it, call it again.")
        result = await self.jobs.dispatch(label, fn, background=False)
        await asyncio.sleep(SETTLE_S)
        self.camera.moved()
        await self.camera.after_move()
        self._last_move = (label, time.monotonic())
        return result

    def _refuse(self, label, reason):
        self.telemetry.tool_end(self.telemetry.tool_start(label), reason, ok=False)
        return reason

    @function_tool()
    async def drive_forward(self, centimeters: int) -> str:
        """Drive straight forward, smoothly. Slows down and stops early if something is in the way.

        Args:
            centimeters: distance to drive, 5 to 300.
        """
        cm = max(5, min(300, centimeters))
        return await self._drive(f"Drive forward {cm} cm", lambda r: _forward(r, cm))

    @function_tool()
    async def drive_back(self, centimeters: int) -> str:
        """Back up straight. There is no sensor behind, so keep it short.

        Args:
            centimeters: distance to back up, 5 to 40.
        """
        cm = max(5, min(40, centimeters))
        return await self._drive(f"Back up {cm} cm", lambda r: _back(r, cm))

    @function_tool()
    async def turn(self, direction: Literal["left", "right"], degrees: int) -> str:
        """Spin the whole car in place.

        Args:
            direction: which way to spin, from the car's point of view.
            degrees: how far to spin, 5 to 180.
        """
        deg = max(5, min(180, abs(degrees))) * (1 if direction == "left" else -1)
        return await self._drive(f"Turn {direction} {abs(deg)}°", lambda r: _turn(r, deg))

    @function_tool()
    async def explore(self, seconds: int) -> str:
        """Roam without a specific goal, avoiding obstacles.

        Returns at once while the car keeps driving. You are told when it ends.

        Args:
            seconds: how long to explore, 10 to 120.
        """
        secs = max(10, min(120, seconds))
        return await self.jobs.dispatch(f"exploring for {secs} seconds", lambda r: _explore(r, secs))

    @function_tool()
    async def look_direction(self, direction: Literal["left", "right", "forward"], degrees: int = 45) -> str:
        """Pan only the head (camera and distance sensor); the car stays still.

        Args:
            direction: where to point the head, from the car's point of view.
            degrees: how far from straight ahead, 10 to 80. Ignored for forward.
        """
        deg = 0 if direction == "forward" else max(10, min(80, abs(degrees))) * (1 if direction == "left" else -1)
        label = "Look forward" if deg == 0 else f"Look {direction} {abs(deg)}°"
        result = await self.jobs.dispatch(label, lambda r: _look(r, deg), background=False)
        await asyncio.sleep(SETTLE_S)
        self.camera.moved()
        await self.camera.after_move()
        return result

    @function_tool()
    async def scan_distances(self) -> str:
        """Sweep the head left to right and measure free space in each direction, in cm."""
        return await self.jobs.dispatch("Scan distances", _scan, background=False)

    @function_tool()
    async def follow_person(self, distance_cm: int = 85) -> str:
        """Follow a visible person, keeping roughly distance_cm away, and report if lost.

        Args:
            distance_cm: desired separation, 65 to 120 cm.
        """
        cm = max(65, min(120, distance_cm))
        return await self.jobs.dispatch("following a person", lambda r: follow_person(r, self.jobs.vision, cm))

    @function_tool()
    async def patrol(self, seconds: int = 30) -> str:
        """Roam, save pictures, and report when done.

        Args:
            seconds: desired duration, 10 to 120 seconds.
        """
        secs = max(10, min(120, seconds))
        return await self.jobs.dispatch(f"a {secs} second patrol", lambda r: patrol(r, self.jobs.vision, secs))

    @function_tool()
    async def find_object(self, name: str) -> str:
        """Search for a named visible object, then approach safely and report.

        Args:
            name: concise visual name of a physical object, not a named place.
        """
        name = " ".join(name.split())[:100]
        if not name:
            return "Tell me what object to look for."
        loop = asyncio.get_running_loop()
        return await self.jobs.dispatch(
            f"searching for {name}",
            lambda r: find_object(r, self.jobs.vision, lambda jpeg: self.object_model.locate(name, jpeg), name, loop),
        )

    @function_tool()
    async def task_status(self) -> str:
        """Check the current or most recent task without ending it."""
        return self.jobs.status()

    @function_tool()
    async def stop(self) -> str:
        """Stop all motion immediately."""
        return await self.jobs.stop("I was told to stop")


server = AgentServer()


class PaneActions:
    def __init__(self, session, telemetry, jobs):
        self.session = session
        self.telemetry = telemetry
        self.jobs = jobs

    async def estop(self):
        await self.session.interrupt(force=True)

    def set_autonomy(self, on):
        self.telemetry.agent["autonomy"] = on
        if not on:
            self.jobs.cancel("the human paused autonomy")

    def set_initiative(self, on):
        self.telemetry.agent["initiative"] = on
        if not on:
            self.jobs.cancel("initiative was switched off", initiative_only=True)

    def set_mic(self, on):
        self.session.input.set_audio_enabled(on)
        self.telemetry.agent["mic"] = on
        self.telemetry.touch()

    def say(self, text):
        self.session.current_agent.new_user_turn()
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
    telemetry.agent["initiative"] = INITIATIVE
    car = Car(HOST, PORT, telemetry)
    camera = CarCamera(CAMERA_URL, STREAM_URL, telemetry)
    session = make_session()
    _follow_session(session, telemetry)
    agent = CarAgent(car, camera)
    await session.start(room=ctx.room, agent=agent)
    if CAR_VOICE == "gemini-live":
        session.input.video = LiveVideo(camera)
    camera.start()
    agent.jobs.start()
    pane = Dashboard(telemetry, Pilot(car, telemetry, MANUAL_SPEED, behaviors.scan),
                     PaneActions(session, telemetry, agent.jobs), None, DASHBOARD_PORT)
    await pane.start()

    async def shutdown():
        await pane.close()
        await agent.jobs.close()
        camera.close()
        await asyncio.to_thread(car.close)

    ctx.add_shutdown_callback(shutdown)
    await session.generate_reply(instructions="Say a short hello and that you can look around, find people, and find things.")


if __name__ == "__main__":
    agents.cli.run_app(server)
