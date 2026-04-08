# PI RPLidar A1 Scan Code V1
# Jacob Thomas
# Team L29 
# 2026

# All in one implementation for ease of testing on remote desktop rasberry pi

import os
import sys
import time
import math
from pathlib import Path

import serial
import RPi.GPIO as GPIO

# PyRPLIDAR Master
SYNC_BYTE1 = 0xA5
SYNC_BYTE2 = 0x5A
CMD_STOP = 0x25
CMD_RESET = 0x40
CMD_FORCE_SCAN = 0x21
DESCRIPTOR_LEN = 7
NODE_LEN = 5

LIDAR_PORT = "/dev/ttyUSB0"
LIDAR_BAUD = 115200

# GPIO Pins
ENABLE = 17 # active low
STEP   = 24
DIR    = 25
MS1    = 27
MS2    = 22
LIMIT = 7 # active high

# Default output directory when P2BP_SCAN_OUTPUT_XYZ is not set (manual runs).
OUT_DIR = os.getenv("P2BP_SCAN_WORKDIR", ".").strip() or "."


def _noninteractive() -> bool:
    """True when run from scan_orchestrator/systemd (no TTY)."""
    v = os.getenv("P2BP_LIDAR_SCAN_NONINTERACTIVE", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    try:
        return not sys.stdin.isatty()
    except Exception:
        return True


def _workdir() -> str:
    return (os.getenv("P2BP_SCAN_WORKDIR") or OUT_DIR or ".").strip() or "."


def _lidar_port() -> str:
    return (os.getenv("P2BP_LIDAR_PORT") or LIDAR_PORT).strip() or LIDAR_PORT


def _pause_interactive_or_sleep(prompt: str, seconds: float = 2.0) -> None:
    if _noninteractive():
        print(f"[non-interactive] {prompt.strip()} (pause {seconds:.1f}s)")
        time.sleep(seconds)
    else:
        input(prompt)

MIN_DIST_MM = 150.0
MAX_DIST_MM = 13000.0

SLICES = 360
CAPTURE_WINDOW = 0.08
LIDAR_WARMUP = 1.0
STEP_DELAY = 0.001
DIR_SETTLE = 0.002
MICROSTEP_DIV = 8
MIN_STEPS_BETWEEN_EDGES = 500
GUARD_MAX_STEPS = 2000000

HOME_TIMEOUT = 60.0
REV_TIMEOUT  = 120.0
EDGE_DEBOUNCE = 0.02

class SimpleRPLidarA1:

    def __init__(self, port = LIDAR_PORT, baud = LIDAR_BAUD, timeout = 1.0):
        self.ser = serial.Serial(port, baud, timeout = timeout)
        self.setMotor(False) # Broken, depeneding on init time
        self.scanning = False

    def close(self):
        try:
            self.stop()
        except Exception:
            pass
        try:
            self.setMotor(False)
        except Exception:
            pass
        try:
            self.ser.close()
        except Exception:
            pass

    def setMotor(self, on: bool):
        # True  => off / False => on
        try:
            self.ser.setDTR(True if (not on) else False)
        except Exception:
            try:
                self.ser.set_dtr(True if (not on) else False)
            except Exception:
                pass

    def clearBuffer(self):
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass

    def writeSerCMD(self, cmd: int):
        self.ser.write(bytes([SYNC_BYTE1, cmd]))

    def stop(self):
        self.writeSerCMD(CMD_STOP)
        time.sleep(0.05)
        self.scanning = False
        self.clearBuffer()

    def reset(self):
        self.writeSerCMD(CMD_RESET)
        time.sleep(0.5)
        self.clearBuffer()

    def readDescriptor(self) -> bytes:
        desc = self.ser.read(DESCRIPTOR_LEN)
        if len(desc) != DESCRIPTOR_LEN:
            raise RuntimeError(f"Descriptor read short ({len(desc)}/{DESCRIPTOR_LEN})")
        if desc[0] != SYNC_BYTE1 or desc[1] != SYNC_BYTE2:
            raise RuntimeError(f"Bad descriptor sync: {hex(desc[0])}, {hex(desc[1])}")
        return desc

    def startScan(self, attempts: int = 12):

        lastErr = None

        for i in range(1, attempts + 1):
            try:
                # hard stop and clear
                try:
                    self.stop()
                except Exception:
                    pass
                self.clearBuffer()

                # reset on first attempt
                if i == 1 or i % 4 == 0:
                    try:
                        self.reset()
                    except Exception:
                        pass

                # motor on
                self.setMotor(True)
                time.sleep(0.15)
                self.clearBuffer()

                # start scan
                self.writeSerCMD(CMD_FORCE_SCAN)

                desc = self.readDescriptor()
                dataLen = desc[2] | (desc[3] << 8) | (desc[4] << 16) | ((desc[5] & 0x3F) << 24)
                sendMode = (desc[5] >> 6) & 0x03
                dataType = desc[6]
                print(f"Descriptor OK: Data Len = {dataLen}, Data Type = 0x{dataType:02x}, Send Mode = {sendMode}")

                self.scanning = True

                # Resync to valid node
                self.resyncNode(maxBytes = 8000)

                # Confirm
                mGood = 0
                t0 = time.time()
                while time.time() - t0 < 0.8:
                    m = self.readMeasurement()
                    if m is not None:
                        mGood += 1
                        if mGood >= 5:
                            break

                if mGood < 1:
                    raise RuntimeError("Started scan but no valid measurements")

                return

            except Exception as e:
                lastErr = e
                try:
                    self.clearBuffer()
                except Exception:
                    pass
                time.sleep(0.2)

        raise RuntimeError(f"Failed to start Scan after {attempts} attempts, Last error: {lastErr}")

    @staticmethod
    def validNode(b0: int, b1: int, b2: int, b3: int, b4: int) -> bool:

        s = (b0 & 0x01)
        sInv = (b0 >> 1) & 0x01
        if (s ^ sInv) != 1:
            return False
        if (b1 & 0x01) != 0x01:
            return False
        return True

    def resyncNode(self, maxBytes: int = 5000):

        buf = bytearray()
        for i in range(maxBytes):
            b = self.ser.read(1)
            if not b:
                continue
            buf += b
            if len(buf) < NODE_LEN:
                continue
            if len(buf) > NODE_LEN:
                buf = buf[-NODE_LEN:]

            b0, b1, b2, b3, b4 = buf
            if self.validNode(b0, b1, b2, b3, b4):
                # aligned
                self.preFetch = bytes(buf)
                return

        raise RuntimeError("Could not resync")

    def readMeasurement(self):

        if hasattr(self, "preFetch") and self.preFetch is not None:
            raw = self.preFetch
            self.preFetch = None
        else:
            raw = self.ser.read(NODE_LEN)

        if len(raw) != NODE_LEN:
            return None

        b0, b1, b2, b3, b4 = raw

        if not self.validNode(b0, b1, b2, b3, b4):
            return None

        quality = (b0 >> 2) & 0x3F

        # angle q6 = (b1 >> 1) | (b2 << 7), then / 64
        angleQ6 = ((b1 >> 1) | (b2 << 7)) & 0xFFFF
        angleDeg = angleQ6 / 64.0

        # distance q2 = b3 | (b4 << 8), then / 4
        distQ2 = (b3 | (b4 << 8)) & 0xFFFF
        distMm = distQ2 / 4.0

        return angleDeg, distMm, quality


# Micro Step Setting
def setMicroStepping(divisor: int) -> None:

    table = {
        8:  (GPIO.LOW,  GPIO.LOW),
        16: (GPIO.HIGH, GPIO.LOW),
        32: (GPIO.LOW,  GPIO.HIGH),
        64: (GPIO.HIGH, GPIO.HIGH),
    }

    if divisor not in table:
        raise ValueError(f"Unsupported divisor {divisor}, Use one of: {list(table.keys())}")
    
    a, b = table[divisor]
    GPIO.output(MS1, a)
    GPIO.output(MS2, b)
    time.sleep(0.01)


# Stepper Motor Control
def enableSteperMotor(on: bool) -> None:
    GPIO.output(ENABLE, GPIO.LOW if on else GPIO.HIGH)
    time.sleep(0.01)

def stepSteperMotor(nSteps: int, direction: int) -> None:
    setDir(direction)
    for i in range(nSteps):
        stepOnce()


def setDir(level: int) -> None:
    GPIO.output(DIR, level)
    time.sleep(DIR_SETTLE)


def stepOnce() -> None:
    GPIO.output(STEP, GPIO.HIGH)
    time.sleep(STEP_DELAY)
    GPIO.output(STEP, GPIO.LOW)
    time.sleep(STEP_DELAY)


# Limit Home Switch
def pressed() -> bool:
    return GPIO.input(LIMIT) == GPIO.HIGH


def homeToPressed(searchDir: int) -> None:
    if pressed():
        setDir(GPIO.HIGH if searchDir == GPIO.LOW else GPIO.LOW)
        t0 = time.time()
        while pressed():
            if time.time() - t0 > HOME_TIMEOUT:
                raise TimeoutError("Timeout releasing before homing")
            stepOnce()

    setDir(searchDir)
    t0 = time.time()
    while True:
        if time.time() - t0 > HOME_TIMEOUT:
            raise TimeoutError("Homing timeout")
        stepOnce()
        if pressed():
            time.sleep(EDGE_DEBOUNCE)
            if pressed():
                print("Home found (switch pressed)")
                return


def findStartFallingEdge(forwardDir: int) -> None:
    setDir(forwardDir)
    prev = GPIO.input(LIMIT)
    t0 = time.time()

    while True:
        if time.time() - t0 > 10.0:
            raise TimeoutError("Could not find falling edge leaving home")
        stepOnce()
        curr = GPIO.input(LIMIT)

        if prev == GPIO.HIGH and curr == GPIO.LOW:
            time.sleep(EDGE_DEBOUNCE)
            if GPIO.input(LIMIT) == GPIO.LOW:
                print("Start reference at falling edge (switch released)")
                return

        prev = curr


def measureNthFallingEdge(forwardDir: int, stopFallingEdgeN: int) -> int:
    setDir(forwardDir)

    steps = 0
    fallingEdgesSeen = 0
    lastEdgeStep = -10**9

    prev = GPIO.input(LIMIT)
    t0 = time.time()

    while True:
        if time.time() - t0 > REV_TIMEOUT:
            raise TimeoutError("Revolution timeout")
        if steps >= GUARD_MAX_STEPS:
            raise RuntimeError("Max steps exceeded")

        stepOnce()
        steps += 1
        curr = GPIO.input(LIMIT)

        if prev == GPIO.HIGH and curr == GPIO.LOW:
            time.sleep(EDGE_DEBOUNCE)
            if GPIO.input(LIMIT) == GPIO.LOW:
                if steps - lastEdgeStep >= MIN_STEPS_BETWEEN_EDGES:
                    fallingEdgesSeen += 1
                    lastEdgeStep = steps
                    print(f"Falling edge #{fallingEdgesSeen} at step {steps}")

                    if fallingEdgesSeen == stopFallingEdgeN:
                        return steps

        prev = curr


# Menu
def askYN(prompt: str) -> bool:
    while True:
        ans = input(prompt).strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer y/n")

def returnStartEdge(searchDir: int, forwardDir: int) -> None:

    print("Returning to origin using home falling edge")
    homeToPressed(searchDir)
    findStartFallingEdge(forwardDir)
    print("Aligned to start falling edge origin")


# Coordinate math
def polarToCartesian(lidarAngleDeg: float, yawDeg: float, distMm: float):

    r = distMm / 1000.0
    theta = math.radians(lidarAngleDeg)
    yaw = math.radians(yawDeg)

    x = r * math.cos(yaw) * math.sin(theta)
    y = r * math.sin(yaw) * math.sin(theta)
    z = r * math.cos(theta)
    return x, y, z


# 3D Scan
def capture3dPointcloud(lidar: SimpleRPLidarA1, stepsPerRev: int, slices: int):
    forced = (os.getenv("P2BP_SCAN_OUTPUT_XYZ") or "").strip()
    if forced:
        xyzPath = forced
        Path(xyzPath).parent.mkdir(parents=True, exist_ok=True)
    else:
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        wd = _workdir()
        os.makedirs(wd, exist_ok=True)
        xyzPath = os.path.join(wd, f"rplidar_3d_{ts}.xyz")

    stepsPerSlicef = stepsPerRev / float(slices)
    yawDeg = 0.0
    totalWritten = 0

    with open(xyzPath, "w") as f:
        for i in range(slices):
            raw = 0
            wrote = 0

            t0 = time.time()
            while (time.time() - t0) < CAPTURE_WINDOW:
                m = lidar.readMeasurement()
                if m is None:
                    continue
                ang, dist, q = m
                raw += 1

                if dist <= 0:
                    continue
                if dist < MIN_DIST_MM or dist > MAX_DIST_MM:
                    continue

                x, y, z = polarToCartesian(ang, yawDeg, dist)
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
                wrote += 1

            totalWritten += wrote
            print(f"Slice {i:03d} yaw = {yawDeg:7.2f}° raw = {raw} wrote = {wrote}")

            # step to next slice
            stepsThisNext = int(round((i + 1) * stepsPerSlicef) - round(i * stepsPerSlicef))
            if stepsThisNext < 0:
                stepsThisNext = 0

            stepSteperMotor(stepsThisNext, direction = GPIO.LOW)
            yawDeg += 360.0 / slices

    print(f"\n3D scan complete, Wrote {totalWritten} points")
    print(f"XYZ: {xyzPath}")

    # Return to origin in reverse
    print("Returning to origin (reverse)")
    REVERSE_DIR = GPIO.HIGH
    stepSteperMotor(stepsPerRev, direction = REVERSE_DIR)


# Main
def setupGPIO():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    GPIO.setup(ENABLE, GPIO.OUT)
    GPIO.setup(STEP, GPIO.OUT)
    GPIO.setup(DIR, GPIO.OUT)
    GPIO.setup(MS1, GPIO.OUT)
    GPIO.setup(MS2, GPIO.OUT)
    GPIO.setup(LIMIT, GPIO.IN)  # external resistor no internal pulls

    GPIO.output(STEP, GPIO.LOW)
    enableSteperMotor(False)


def calibrateStepsPerRev() -> int:
    raw_steps = (os.getenv("P2BP_STEPS_PER_REV") or "").strip()
    if raw_steps:
        v = int(raw_steps)
        if v > 0:
            print(f"Using P2BP_STEPS_PER_REV={v}")
            return v

    steps_file = (os.getenv("P2BP_STEPS_PER_REV_FILE") or "").strip()
    if not steps_file:
        steps_file = os.path.join(_workdir(), "stepsPerRev.txt")
    try:
        sp = Path(steps_file)
        if sp.is_file():
            line = sp.read_text(encoding="utf-8").strip().split()
            if line:
                v = int(line[0])
                if v > 0:
                    print(f"Using steps/rev from {steps_file}: {v}")
                    return v
    except (ValueError, OSError) as e:
        print(f"Could not read steps file ({steps_file}): {e}")

    # extraBump = askYN("Extra bump between 0 and 360? (y/n): ")
    # stopOnN = 2 if extraBump else 1 self.setMotor(True)
    stopOnN = 1
    print(f"Measuring steps per rev: stop on falling edge #{stopOnN} after start")

    setMicroStepping(MICROSTEP_DIV)

    _pause_interactive_or_sleep("\nPress Enter to enable motor and start calibration\n", 3.0)
    enableSteperMotor(True)

    SEARCH_DIR  = GPIO.HIGH
    FORWARD_DIR = GPIO.LOW

    homeToPressed(SEARCH_DIR)
    findStartFallingEdge(FORWARD_DIR)
    stepsPerRev = measureNthFallingEdge(FORWARD_DIR, stopFallingEdgeN = stopOnN)

    print(f"Steps per 360°: {stepsPerRev}")
    print(f"Steps/deg: {stepsPerRev / 360.0:.6f}")

    # Return to origin in reverse
    print("Returning to origin (reverse)")
    REVERSE_DIR = GPIO.HIGH if FORWARD_DIR == GPIO.LOW else GPIO.LOW
    stepSteperMotor(stepsPerRev, direction = REVERSE_DIR)

    print("Returned to origin")

    steps_out = os.path.join(_workdir(), "stepsPerRev.txt")
    _parent = os.path.dirname(steps_out)
    if _parent:
        os.makedirs(_parent, exist_ok=True)
    with open(steps_out, "w", encoding="utf-8") as f:
        f.write(str(stepsPerRev) + "\n")
    print(f"Saved to {steps_out}")

    return stepsPerRev


def main():

    setupGPIO()
    lidar = None

    try:
        stepsPerRev = calibrateStepsPerRev()

        _pause_interactive_or_sleep("\nPress Enter to start 3D scan\n", 1.0)

        # Connect to lidar
        lidar = SimpleRPLidarA1(_lidar_port(), LIDAR_BAUD, timeout = 1.0)
        print("PyRPlidar: device is connected")

        # Start scan
        lidar.startScan(attempts = 12)

        # Warmup discard
        if LIDAR_WARMUP > 0:
            print(f"Warming up lidar for {LIDAR_WARMUP:.1f}s")
            t0 = time.time()
            while time.time() - t0 < LIDAR_WARMUP:
                lidar.readMeasurement()

        # Enable stepper motor
        enableSteperMotor(True)

        capture3dPointcloud(lidar, stepsPerRev, slices = SLICES)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        try:
            if lidar is not None:
                lidar.close()
                print("PyRPlidar: device is disconnected")
        except Exception:
            pass

        try:
            enableSteperMotor(False)
        except Exception:
            pass

        GPIO.cleanup()
        print("GPIO cleaned up")


if __name__ == "__main__":
    main()
