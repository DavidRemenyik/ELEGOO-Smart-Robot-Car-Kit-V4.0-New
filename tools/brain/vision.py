"""Independent local JPEG/person loop; YOLOX-Nano COCO ONNX, CPU, Apache-2.0."""

import logging
import io
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

from PIL import Image

MODEL = Path(sys.prefix) / "share/elegoo/yolox_nano.onnx"
SIZE = 416
MAX_JPEG = 2_000_000
log = logging.getLogger("car-agent")


@dataclass(frozen=True, slots=True)
class Person:
    bearing: float  # camera-relative degrees, positive left
    width: float  # fraction of image width
    height: float
    clipped: bool


@dataclass(frozen=True, slots=True)
class Frame:
    timestamp: float  # capture request start, monotonic
    jpeg: bytes
    people: tuple[Person, ...]


def person_box(box: tuple[float, float, float, float], hfov: float) -> Person:
    x0, y0, x1, y1 = box
    bearing = math.degrees(math.atan((1 - x0 - x1) * math.tan(math.radians(hfov / 2))))
    return Person(bearing, x1 - x0, y1 - y0, x0 < 0.015 or x1 > 0.985)


class Detector:
    def __init__(self, model: Path = MODEL):
        import numpy as np
        import onnxruntime as ort

        self.np = np
        options = ort.SessionOptions()
        options.intra_op_num_threads = 3
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        grids, strides = [], []
        for stride in (8, 16, 32):
            y, x = np.mgrid[: SIZE // stride, : SIZE // stride]
            grids.append(np.stack((x, y), axis=-1).reshape(-1, 2))
            strides.append(np.full((x.size, 1), stride))
        self.grid, self.stride = np.concatenate(grids), np.concatenate(strides)

    def detect(self, jpeg: bytes) -> tuple[Person, ...]:
        """Return visible person boxes projected into camera bearings."""
        np = self.np
        with Image.open(io.BytesIO(jpeg)) as image:
            w, h = image.size
            ratio = SIZE / max(h, w)
            resized = np.asarray(image.convert("RGB").resize((int(w * ratio), int(h * ratio)), Image.Resampling.BILINEAR))
        padded = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
        padded[: resized.shape[0], : resized.shape[1]] = resized[:, :, ::-1]  # YOLOX expects BGR
        blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
        prediction = self.session.run(None, {self.input: blob})[0][0]
        scores = prediction[:, 4] * prediction[:, 5]  # first COCO class is person
        keep = scores >= 0.30
        predictions = prediction[keep]
        centers = (predictions[:, :2] + self.grid[keep]) * self.stride[keep]
        sizes = np.exp(predictions[:, 2:4]) * self.stride[keep]
        low, high = centers - sizes / 2, centers + sizes / 2
        areas = sizes.prod(axis=1)
        order = np.argsort(-scores[keep])
        people = []
        while order.size:
            i, rest = order[0], order[1:]
            overlap = np.maximum(0, np.minimum(high[i], high[rest]) - np.maximum(low[i], low[rest])).prod(axis=1)
            order = rest[overlap / (areas[i] + areas[rest] - overlap) <= 0.45]
            x, y = low[i]
            right, bottom = high[i]
            x0, y0, x1, y1 = np.clip(
                (x / (w * ratio), y / (h * ratio), right / (w * ratio), bottom / (h * ratio)), 0, 1
            )
            if x1 > x0 and y1 > y0:
                people.append(person_box((x0, y0, x1, y1), 52))
        return tuple(people)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Vision:
    """Publish only the latest frame; never queue images or read the control link."""

    def __init__(self, camera_url: str):
        self.url = camera_url
        self.fake = (
            os.getenv("CAR_FAKE_VISION") == "1"
            and urlsplit(camera_url).hostname in ("127.0.0.1", "::1")
        )
        self._frame: Frame | None = None
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="car-vision")
            self._thread.start()

    def latest(self) -> Frame | None:
        frame = self._frame
        return frame if frame and time.monotonic() - frame.timestamp < 0.55 else None

    def close(self) -> None:
        self._closed.set()
        if self._thread:
            self._thread.join(timeout=1.2)

    def _run(self) -> None:
        if not self.fake and not MODEL.is_file():
            log.error("Person detector not installed: %s (the download command is at the top of agent.py)", MODEL)
            return
        detector = None if self.fake else Detector()
        opener = build_opener(_NoRedirect)
        failing = False
        while not self._closed.is_set():
            started = time.monotonic()
            try:
                with opener.open(self.url, timeout=1) as response:
                    jpeg = response.read(MAX_JPEG + 1)
                    fixture = response.headers.get("X-Fake-Person", "")
                if not jpeg or len(jpeg) > MAX_JPEG:
                    raise ValueError("unexpected JPEG size")
                if detector:
                    people = detector.detect(jpeg)
                elif fixture:
                    x0, y0, x1, y1 = map(float, fixture.split(","))
                    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                        raise ValueError("invalid stand-in box")
                    people = (person_box((x0, y0, x1, y1), 60),)
                else:
                    people = ()
                self._frame = Frame(started, jpeg, people)
                failing = False
            except (ValueError, OSError) as e:
                self._frame = None
                if not failing:
                    log.warning("Person camera unavailable: %s", e)
                failing = True
                self._closed.wait(0.5)
            self._closed.wait(max(0, 1 / 12 - (time.monotonic() - started)))
