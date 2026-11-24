"""Closed-loop driving, run on the Mac at the pace of the car's link (about 10 to 15 commands a second).

Every tick sends signed wheel speeds and gets the gyro heading and floor state back, so the car holds
its heading on straights, turns to a measured angle, and ramps its speed instead of stop-starting.
Distance is dead reckoning from a calibrated speed: the car has no wheel encoders, so a StallWatch
checks that the camera picture really changes while the wheels drive straight.
"""

import io
import json
import logging
import statistics
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

import behaviors
from robot import MAX_RANGE_CM

log = logging.getLogger("car-agent")

CALIBRATION = Path(__file__).with_name("calibration.json")
CRUISE = 150  # wheel PWM on a clear straight
CREEP = 80  # slowest PWM that still moves the car
DEAD = 12  # PWM where the measured speed line crosses zero
BACK = 110
RAMP = 300  # PWM per second
STEER = 4.0  # PWM per degree of heading error
STEER_MAX = 60
SLOW_CM = 45  # the car eases off over this much remaining room
SONAR_EVERY = 3  # ticks between range readings while driving forward
TURN_MIN, TURN_MAX = 65, 110  # measured: 80 spins about 130 deg/s, and the gyro saturates at 250 deg/s
TURN_GAIN = 0.5  # PWM per degree left to turn
TURN_DONE = 3.0
TURN_RETRY = 6.0  # a turn that settles further off than this gets one correcting pass
TURN_LEAD_S = 0.15  # the car keeps spinning about this long after the stop command
NO_FLOOR = "no floor under my front sensors (an edge, or I am lifted)"
MOVE_LIMIT = "that is as far as I go in one move"
STALLED = "I was pushing against something: my wheels turned but my camera view stood still"

# Stall detection. While the wheels drive straight, three independent signs of not moving each vote:
#   picture: the camera view hardly changes (after allowing for the turn the gyro and head report);
#   gyro:    the heading creeps although the command says straight, or heading hold sits at its limit;
#   range:   the range reading stays the same (a reading that shrinks the way the car drives says it moves;
#            one that jitters without a trend says nothing).
# Two votes over STALL_WINDOW_S make a stall. A sign that cannot be read abstains: a dark or blank picture,
# deliberate steering, a range reading at its cap. When only one sign can tell (the gyro mostly cannot
# while heading hold works), that sign saying still for ONE_SIGN_S is enough, unless the other readable
# sign says the car moves. With no fresh camera frame nothing is judged at all.
# Values marked GUESS come from the building stand-in (a depth-shaded render, not a real camera) and
# fake_car.py; set them from stall_probe.py recordings on the real car. Measurements: .omo/waves/p3/stall.md.
STALL_MIN_PWM = 80  # straight drives slower than this are not judged (CREEP is 80)
STALL_WINDOW_S = 1.2  # the votes must hold this long
ONE_SIGN_S = 2.5  # GUESS: a still picture, or a steady range reading, this long with no sign of motion
STALL_MIN_FRAMES = 5  # fresh frames needed inside the window (the real camera gives about 11 a second)
STALL_FRAME_AGE_S = 0.4  # with no frame newer than this the camera counts as stalled
STILL_SCORE = 1.0  # GUESS: picture change / texture; stand-in stuck 0.03-0.54, moving 3.9-13.8 (5-95 %)
TEXTURE_MIN = 1.0  # GUESS: mean grey step between pixels; under it the view is blank (stand-in moving: 5 % at 1.25)
DARK_MIN = 20  # GUESS: mean grey level; under it the room is too dark to judge
GYRO_CREEP_DEG = 1.5  # GUESS: heading change in the window; under heading hold the stand-in stays within 0.4
RANGE_STILL_CM = 3  # GUESS: spread of the range readings; stand-in noise 0.3 cm, real sensors about 1 cm
RANGE_MIN_READINGS = 3  # range readings needed in the window (forward() reads one every third tick)
STALL_BACK_CM = 15  # recovery: back off this far, then turn away from straight ahead
STALL_AVOID_DEG = 30  # recovery ignores scan directions this close to straight ahead
PICTURE = (80, 60)  # the JPEG is decoded at 1/8 scale, grey
PX_PER_DEG = 80 / 52  # at PICTURE width, 52 degree camera
SHIFT_X, SHIFT_Y = 4, 2  # pixels searched around the gyro-predicted shift

def _calibrated_speed():
    try:
        return float(json.loads(CALIBRATION.read_text())["cm_per_s"])
    except FileNotFoundError:
        return 55.0  # from 27 cm/s at PWM 80 and 39 cm/s at 110, measured on carpet


CM_PER_S = _calibrated_speed()  # at CRUISE


def _speed(pwm):
    return CM_PER_S * max(0.0, abs(pwm) - DEAD) / (CRUISE - DEAD)


def picture(jpeg):
    """The JPEG as a small grey array, decoded at 1/8 scale so it costs about a millisecond."""
    with Image.open(io.BytesIO(jpeg)) as image:
        image.draft("L", PICTURE)
        grey = image.convert("L")
        if grey.size != PICTURE:
            grey = grey.resize(PICTURE, Image.Resampling.BILINEAR)
        return np.asarray(grey, dtype=np.float32)


def texture(grey):
    """Mean grey-level step between neighbouring pixels: how much a picture could show motion."""
    return float(np.abs(np.diff(grey, axis=1)).mean() + np.abs(np.diff(grey, axis=0)).mean()) / 2


def change(before, after, shift=0.0):
    """Mean grey difference between two pictures, at the best alignment near `shift` pixels to the right.

    The search absorbs turning (the view slides sideways) and bobbing; driving makes things grow and
    spread outwards, which no shift explains. Mean brightness is removed, so exposure changes do not count.
    """
    h, w = before.shape
    centre = round(max(-20.0, min(20.0, shift)))
    mx = abs(centre) + SHIFT_X
    a = before[SHIFT_Y:h - SHIFT_Y, mx:w - mx]
    a = a - a.mean()
    best = float("inf")
    for dy in range(-SHIFT_Y, SHIFT_Y + 1):
        for dx in range(centre - SHIFT_X, centre + SHIFT_X + 1):
            b = after[SHIFT_Y + dy:h - SHIFT_Y + dy, mx + dx:w - mx + dx]
            best = min(best, float(np.abs(a - (b - b.mean())).mean()))
    return best


def distance_at(marks, since):
    """From (time, distance so far) marks, the distance reached by `since`, when the car came to rest.

    The latest mark at or before `since`, not the largest: a running count drops at every earlier stall.
    """
    return next((gone for at, gone in reversed(marks) if at <= since), marks[0][1] if marks else 0.0)


class StallWatch:
    """Notices a straight drive that goes nowhere: two of picture, gyro and range say the car is not moving.

    The robot wrapper that sends the wheel commands feeds it (wheels() after each command, ranged() after
    each range reading); drives read stalled(). While the picture feed has no fresh frame, gyro and range
    still vote. Its votes are logged once a second during straight drives.
    """

    def __init__(self, camera=None):
        self.camera = camera  # anything with newest(max_age) -> Frame | None, such as vision.Vision
        self.events = []  # (found at, still since) per stall, monotonic seconds
        self.signal = {}  # the latest raw values and votes, for logs
        self._logged_at = 0.0
        self._clear()

    def _clear(self):
        self._sign, self._began, self._newest = 0, 0.0, 0.0
        self._pictures = deque()  # (frame timestamp, grey picture, view direction in degrees)
        self._commands = deque()  # (time, heading, steering = (right - left) / 2)
        self._ranges = deque()  # (time, cm) with the head facing ahead
        self._picture_vote, self._picture_since, self._range_since = None, None, None
        self.still_since = None

    def stalled(self):
        return STALLED if self.still_since is not None else None

    def ranged(self, cm, head):
        if self._sign and abs(head - behaviors.CENTER) <= 15:
            self._ranges.append((time.monotonic(), cm))

    def wheels(self, left, right, heading, head):
        """Judge after a wheel command; True when a stall is newly found. Only straight drives count."""
        sign = (left > 0 and right > 0) - (left < 0 and right < 0)
        if not sign or abs(left + right) / 2 < STALL_MIN_PWM:
            self._clear()
            return False
        now = time.monotonic()
        if sign != self._sign:
            self._clear()
            self._sign, self._began = sign, now
        if self.still_since is not None or self.camera is None:
            return False
        frame = self.camera.newest(STALL_FRAME_AGE_S)
        if frame is None:  # the picture feed has stalled (it does on the real car): gyro and range still vote
            self._pictures.clear()
            self._picture_vote, self._picture_since = None, None
        self._commands.append((now, heading, (right - left) / 2))
        while len(self._commands) > 1 and self._commands[1][0] <= now - STALL_WINDOW_S:
            self._commands.popleft()
        while self._ranges and self._ranges[0][0] < now - STALL_WINDOW_S:
            self._ranges.popleft()
        if frame is not None and frame.timestamp > self._newest and frame.timestamp >= self._began:
            self._newest = frame.timestamp
            self._look(picture(frame.jpeg), frame.timestamp, heading + head)
        if now - self._began < STALL_WINDOW_S:
            return False
        votes = (self._picture_vote, self._gyro(heading), self._range())
        self.signal["votes"] = votes
        if now - self._logged_at >= 1.0:
            self._logged_at = now
            log.info("stall votes picture/gyro/range %s, range spread %s closed %s, picture %s", votes,
                     self.signal.get("spread"), self.signal.get("closed"), "fresh" if frame else "none")
        if not votes[2]:
            self._range_since = None
        elif self._range_since is None:
            self._range_since = now - STALL_WINDOW_S
        lone = [since for since, other in ((self._picture_since, votes[2]), (self._range_since, votes[0]))
                if since is not None and other is not False and now - since >= ONE_SIGN_S]
        if sum(vote is True for vote in votes) >= 2:
            self.still_since = now - STALL_WINDOW_S
        elif lone:
            self.still_since = min(lone)
        else:
            return False
        self.events.append((now, self.still_since))
        return True

    def _look(self, grey, at, view):
        self._pictures.append((at, grey, view))
        while len(self._pictures) > 1 and self._pictures[1][0] <= at - STALL_WINDOW_S:
            self._pictures.popleft()
        since, before, then = self._pictures[0]
        detail, light = texture(before), float(before.mean())
        score = change(before, grey, (view - then) * PX_PER_DEG) / max(detail, 1e-6)
        readable = (at - since >= STALL_WINDOW_S and len(self._pictures) >= STALL_MIN_FRAMES
                    and light >= DARK_MIN and detail >= TEXTURE_MIN)
        self._picture_vote = score < STILL_SCORE if readable else None
        if not self._picture_vote:
            self._picture_since = None
        elif self._picture_since is None:
            self._picture_since = since
        self.signal = dict(t=at, span=at - since, frames=len(self._pictures), score=score, texture=detail,
                           light=light, turned=view - then)

    def _gyro(self, heading):
        steering = [steer for _, _, steer in self._commands]
        if any(abs(steer) > STEER_MAX + 1 for steer in steering):
            return None  # deliberate steering: the gyro cannot tell creep from the turn asked for
        crept = abs(heading - self._commands[0][1])
        held = sum(abs(steer) >= STEER_MAX - 1 for steer in steering) / len(steering)
        self.signal.update(crept=crept, held=held)
        return crept >= GYRO_CREEP_DEG or held >= 0.5

    def _range(self):
        readings = [cm for _, cm in self._ranges]
        if len(readings) < RANGE_MIN_READINGS or max(readings) >= MAX_RANGE_CM:
            return None
        half = len(readings) // 2
        closed = self._sign * (statistics.median(readings[:half]) - statistics.median(readings[half:]))
        self.signal.update(spread=max(readings) - min(readings), closed=closed)
        if max(readings) - min(readings) <= RANGE_STILL_CM:
            return True
        return False if closed >= RANGE_STILL_CM else None


class Driver:
    def __init__(self, robot, should_stop=lambda: None, stall=None):
        self.robot = robot
        self.should_stop = should_stop  # returns a reason to stop now, or None
        self.stall = stall  # the StallWatch the robot wrapper feeds, or None

    def _stop_reason(self):
        return self.should_stop() or (self.stall.stalled() if self.stall else None)

    def _roll(self, pwm, target):
        heading, grounded = self._status
        steer = max(-STEER_MAX, min(STEER_MAX, STEER * (target - heading)))
        self._status = self.robot.wheels(pwm - steer, pwm + steer)

    def _halt(self):
        self._status = self.robot.wheels(0, 0)

    def forward(self, cm, stop_cm, max_s=7.0):
        """Drive straight ahead. Returns (cm driven, reason it stopped early or None)."""
        r = self.robot
        r.head(behaviors.CENTER)
        time.sleep(0.3)
        self._status = r.heading()
        target = self._status[0]
        ahead = r.distance()
        gone, pwm, tick, why = 0.0, 0.0, 0, None
        start = last = time.monotonic()
        marks = deque(maxlen=200)
        try:
            while gone < cm:
                why = self._stop_reason()
                if why == STALLED:
                    gone = distance_at(marks, self.stall.still_since)
                if why:
                    break
                if not self._status[1]:
                    why = NO_FLOOR
                    break
                tick += 1
                if tick % SONAR_EVERY == 0:
                    ahead = r.distance()
                if ahead < stop_cm:
                    why = f"obstacle {ahead} cm ahead"
                    break
                now = time.monotonic()
                if now - start > max_s:
                    why = MOVE_LIMIT
                    break
                gone += _speed(pwm) * (now - last)
                marks.append((now, gone))
                room = min(cm - gone, ahead - stop_cm)
                limit = min(CRUISE, CREEP + (CRUISE - CREEP) * room / SLOW_CM)
                pwm = max(CREEP, min(limit, pwm + RAMP * (now - last)))
                last = now
                self._roll(pwm, target)
        finally:
            self._halt()
        return gone, why

    def back(self, cm):
        """Back up straight; there is no sensor behind. Returns (cm driven, reason it stopped early or None)."""
        self._status = self.robot.heading()
        target = self._status[0]
        gone, why = 0.0, None
        last = time.monotonic()
        marks = deque(maxlen=200)
        try:
            while gone < cm:
                why = self._stop_reason()
                if why == STALLED:
                    gone = distance_at(marks, self.stall.still_since)
                if why:
                    break
                self._roll(-BACK, target)
                now = time.monotonic()
                gone += _speed(BACK) * (now - last)
                marks.append((now, gone))
                last = now
        finally:
            self._halt()
        return gone, why

    def turn(self, degrees, max_s=6.0):
        """Spin in place, positive to the left. Returns (degrees actually turned, reason it stopped early or None)."""
        r = self.robot
        start, _ = r.heading()
        heading, why = start, None
        began = time.monotonic()
        for _ in range(2):
            want = start + degrees - heading
            rate, last = 0.0, time.monotonic()
            try:
                while True:
                    left = start + degrees - heading - rate * TURN_LEAD_S
                    if abs(left) < TURN_DONE or left * want < 0:
                        break
                    why = self.should_stop()
                    if why:
                        break
                    if time.monotonic() - began > max_s:
                        why = "the turn took too long; my wheels may be stuck"
                        break
                    pwm = min(TURN_MAX, TURN_MIN + TURN_GAIN * abs(left))
                    side = 1 if left > 0 else -1
                    new, _ = r.wheels(-side * pwm, side * pwm)
                    now = time.monotonic()
                    rate = (new - heading) / (now - last)
                    heading, last = new, now
            finally:
                r.wheels(0, 0)
            time.sleep(0.2)
            heading, _ = r.heading()
            if why or abs(start + degrees - heading) < TURN_RETRY:
                break
        return heading - start, why

    def explore(self, seconds, stop_cm):
        """Wander: drive until something is in the way, then turn toward the most open direction."""
        deadline = time.monotonic() + seconds
        driven, turns, why = 0.0, 0, None
        while time.monotonic() < deadline:
            gone, why = self.forward(300, stop_cm, max_s=deadline - time.monotonic())
            driven += gone
            if why in (None, MOVE_LIMIT):
                why = None
                continue
            if why == NO_FLOOR:
                self.back(15)
                _, why = self.turn(120)
            elif why.startswith("obstacle"):
                angle, best = max(behaviors.scan(self.robot), key=lambda reading: reading[1])
                _, why = self.turn(180 if best <= stop_cm + 10 else angle - behaviors.CENTER)
            elif why == STALLED:
                # The range sensor missed whatever is ahead, so back off it and pick an open side.
                self.back(STALL_BACK_CM)
                readings = [reading for reading in behaviors.scan(self.robot)
                            if abs(reading[0] - behaviors.CENTER) >= STALL_AVOID_DEG]
                angle, best = max(readings, key=lambda reading: reading[1])
                _, why = self.turn(180 if best <= stop_cm + 10 else angle - behaviors.CENTER)
            else:
                break
            turns += 1
            if why:
                break
        return driven, turns, why


def calibrate(robot, log=print):
    """Check the turn direction against the gyro and measure cruise speed against a wall."""
    start, grounded = robot.heading()
    if not grounded:
        raise SystemExit("put the car on the floor first")
    until = time.monotonic() + 0.3
    while time.monotonic() < until:
        robot.wheels(-80, 80)
    robot.wheels(0, 0)
    time.sleep(0.5)
    turned = robot.heading()[0] - start
    log(f"short spin left: gyro says {turned:+.0f} degrees (the car should have spun counter-clockwise seen from above)")
    if turned < 10:
        raise SystemExit("the gyro did not see a left turn: wheel sides or gyro sign are swapped, fix before driving")
    Driver(robot).turn(-turned)

    robot.head(behaviors.CENTER)
    time.sleep(0.4)
    before = statistics.median(robot.distance() for _ in range(5))
    if not 100 <= before < 150:
        raise SystemExit(f"wall is {before} cm ahead; face a wall 110 to 140 cm away to measure speed")
    driver = Driver(robot)
    driver._status = robot.heading()
    target = driver._status[0]
    began = time.monotonic()
    while time.monotonic() - began < 1.2:
        driver._roll(CRUISE, target)
    took = time.monotonic() - began
    driver._halt()
    time.sleep(0.5)
    after = statistics.median(robot.distance() for _ in range(5))
    speed = (before - after) / took
    log(f"cruise: {before} cm -> {after} cm in {took:.2f} s = {speed:.0f} cm/s")
    if speed <= 5:
        raise SystemExit("the car barely moved; not saving")
    CALIBRATION.write_text(json.dumps({"cm_per_s": round(speed, 1)}) + "\n")
    log(f"saved {CALIBRATION.name}")
