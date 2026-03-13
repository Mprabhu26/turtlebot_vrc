#!/usr/bin/env python3
import os
# Force PulseAudio via WSLg socket before sounddevice loads
os.environ.setdefault('PULSE_SERVER', 'unix:/mnt/wslg/PulseServer')

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import threading
import time
import math
import numpy as np
import sounddevice as sd
import io, re, json

# ─────────────────────────────────────────────
#  WORLD LAYOUT (hospital_vrc.world)
#  Robot spawns at (0,0) facing EAST (+X)
#
#  NW: RECEPTION  (-6, +6)  — orange
#  NE: ICU        (+6, +6)  — yellow
#  SW: WARD       (-6, -6)  — blue
#  SE: PHARMACY   (+6, -6)  — green
#
#  Corridor along y=0, walls at y=+2 and y=-2
#  Gaps at x=-6 and x=+6
# ─────────────────────────────────────────────

SPEED       = 0.9   # m/s linear
TURN_RATE   = 2.0   # rad/s angular
WALL_DIST         = 0.35  # metres — stop manual movement if closer than this
WALL_DIST_NAV     = 0.15  # metres — only stop navigation if really close

SAMPLE_RATE   = 16000
CHUNK_SAMPLES = int(SAMPLE_RATE * 3.0)

# Room centre coordinates
ZONES = {
    'icu':       ( 6.0,  6.0),
    'pharmacy':  ( 6.0, -6.0),
    'reception': (-6.0,  6.0),
    'ward':      (-6.0, -6.0),
}

# ─────────────────────────────────────────────
#  LOCAL FALLBACK NLU
#  Used when Groq is unavailable / quota exceeded
# ─────────────────────────────────────────────
NAVIGATE_KEYWORDS = {
    'icu':        ['icu', 'i c u', 'yellow', 'intensive care', 'intensive'],
    'pharmacy':   ['pharmacy', 'pharma', 'green', 'medicine', 'drug', 'dispensary'],
    'reception':  ['reception', 'orange', 'front desk', 'lobby', 'entrance', 'receptionist'],
    'ward':       ['ward', 'blue', 'patient room', 'room', 'patients'],
}

MOVE_KEYWORDS = {
    'forward':  ['forward', 'go forward', 'move forward', 'go ahead', 'ahead',
                 'straight', 'advance', 'proceed', 'go straight'],
    'backward': ['backward', 'go backward', 'move backward', 'go back', 'back',
                 'reverse', 'retreat', 'back up'],
    'left':     ['turn left', 'rotate left', 'left', 'go left'],
    'right':    ['turn right', 'rotate right', 'right', 'go right'],
    'stop':     ['stop', 'halt', 'freeze', 'cancel', 'pause', 'hold',
                 'enough', 'wait', 'brake'],
    'spin':     ['spin', 'spin around', 'rotate', 'turn around', '360'],
}

def local_nlu(text):
    """Keyword-based fallback NLU. Returns same JSON format as Groq NLU."""
    t = text.lower().strip().rstrip('.')

    # Check stop first (highest priority)
    for kw in MOVE_KEYWORDS['stop']:
        if kw in t:
            return {'action': 'stop'}

    # Check navigation
    for zone, keywords in NAVIGATE_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                return {'action': 'navigate', 'zone': zone}

    # Check movement
    for direction, keywords in MOVE_KEYWORDS.items():
        if direction == 'stop':
            continue
        for kw in keywords:
            if kw in t:
                if direction == 'forward':
                    return {'action': 'move_continuous', 'linear': SPEED, 'angular': 0.0}
                elif direction == 'backward':
                    return {'action': 'move_continuous', 'linear': -SPEED, 'angular': 0.0}
                elif direction == 'left':
                    return {'action': 'move_continuous', 'linear': 0.0, 'angular': TURN_RATE}
                elif direction == 'right':
                    return {'action': 'move_continuous', 'linear': 0.0, 'angular': -TURN_RATE}
                elif direction == 'spin':
                    return {'action': 'move', 'linear': 0.0, 'angular': TURN_RATE,
                            'duration': (2 * math.pi) / TURN_RATE}

    return {'action': 'unknown'}


# ─────────────────────────────────────────────
#  GROQ NLU PROMPT
# ─────────────────────────────────────────────
NLU_PROMPT = f"""You are a hospital robot controller. Extract the command from user speech and return ONLY a JSON object.

Navigation commands (go to a room):
- ICU / yellow room / intensive care → {{"action":"navigate","zone":"icu"}}
- Pharmacy / green room / medicine / dispensary → {{"action":"navigate","zone":"pharmacy"}}
- Reception / orange room / front desk / lobby → {{"action":"navigate","zone":"reception"}}
- Ward / blue room / patient room → {{"action":"navigate","zone":"ward"}}

Continuous movement (runs until stop):
- forward / go forward / move forward / go ahead / advance → {{"action":"move_continuous","linear":{SPEED},"angular":0.0}}
- backward / go back / reverse / retreat → {{"action":"move_continuous","linear":{-SPEED},"angular":0.0}}
- turn left / rotate left / go left → {{"action":"move_continuous","linear":0.0,"angular":{TURN_RATE}}}
- turn right / rotate right / go right → {{"action":"move_continuous","linear":0.0,"angular":{-TURN_RATE}}}

Stop:
- stop / halt / freeze / cancel / wait / hold → {{"action":"stop"}}

Spin (one full rotation):
- spin / spin around / rotate 360 → {{"action":"move","linear":0.0,"angular":{TURN_RATE},"duration":{round((2*math.pi)/TURN_RATE, 2)}}}

Anything else → {{"action":"unknown"}}

Return ONLY the JSON. No explanation, no markdown."""

NOISE = {'gracias','merci','danke','okay','ok','thank you','thanks','yes','no',
         'hmm','uh','um','ah','bye','hello','hi','obrigado','yeah','sure',
         'word','vogue','world','fallen','go to','the','a','an'}


class VoiceControlNode(Node):
    def __init__(self):
        super().__init__('voice_control')

        # Publishers / Subscribers
        self.pub       = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub  = self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.scan_sub  = self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

        # State
        self.x = 0.0; self.y = 0.0; self.yaw = 0.0
        self.front_dist = 999.0          # LiDAR front distance
        self.navigating = False
        self.moving_continuous = False   # for forward/back/turn continuous mode
        self.current_zone = None         # None = unknown position (use odom)
        self.nav_lock = threading.Lock()

        # Groq client
        self.groq_client = None
        self.groq_available = False
        try:
            from groq import Groq
            self.groq_client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
            self.groq_available = True
            self.get_logger().info('Groq ready')
        except Exception as e:
            self.get_logger().warn(f'Groq unavailable: {e} — using local fallback NLU')

        # Faster-Whisper (local ASR fallback)
        self.fw_model = None
        try:
            from faster_whisper import WhisperModel
            self.fw_model = WhisperModel('base', device='cpu', compute_type='int8')
            self.get_logger().info('faster-whisper ready')
        except Exception as e:
            self.get_logger().warn(f'faster-whisper: {e}')

        # VAD
        self.vad_model = None
        try:
            import torch
            model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad',
                                      force_reload=False, trust_repo=True)
            self.vad_model = model
            self.get_logger().info('VAD ready')
        except Exception as e:
            self.get_logger().warn(f'VAD: {e}')

        self.get_logger().info('=== Ready! Say: go to ICU / pharmacy / ward / reception ===')
        self.get_logger().info('=== Or: move forward / turn left / stop ===')
        threading.Thread(target=self._listen_loop, daemon=True).start()

    # ─────────────────────────────────────────
    #  ODOMETRY CALLBACK — track real position
    # ─────────────────────────────────────────
    def _odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)

    # ─────────────────────────────────────────
    #  LIDAR CALLBACK — wall detection
    # ─────────────────────────────────────────
    def _scan_cb(self, msg):
        ranges = msg.ranges
        if not ranges:
            return
        n = len(ranges)
        # Front sector: ±15 degrees
        front_indices = list(range(0, int(n * 15 / 360) + 1)) + \
                        list(range(int(n * 345 / 360), n))
        valid = [r for i in front_indices
                 for r in [ranges[i]]
                 if not math.isinf(r) and not math.isnan(r) and r > 0.05]
        self.front_dist = min(valid) if valid else 999.0

    # ─────────────────────────────────────────
    #  PUBLISH TWIST
    # ─────────────────────────────────────────
    def _pub(self, linear, angular):
        msg = Twist()
        msg.linear.x  = float(linear)
        msg.angular.z = float(angular)
        self.pub.publish(msg)

    def _stop(self):
        self._pub(0.0, 0.0)

    # ─────────────────────────────────────────
    #  CONTINUOUS MOVEMENT (until stop/wall)
    # ─────────────────────────────────────────
    def _start_continuous(self, linear, angular):
        """Run continuous movement in a thread until stop command or wall."""
        def _run():
            self.moving_continuous = True
            self.get_logger().info(f'Continuous move: lin={linear:.1f} ang={angular:.1f} — say STOP to halt')
            while self.moving_continuous:
                # Wall check — only for forward movement
                if linear > 0 and self.front_dist < WALL_DIST:
                    self.get_logger().warn(f'Wall detected ({self.front_dist:.2f}m) — stopping')
                    break
                self._pub(linear, angular)
                time.sleep(0.05)
            self._stop()
            self.moving_continuous = False
            self.get_logger().info('Continuous move stopped')
        threading.Thread(target=_run, daemon=True).start()

    # ─────────────────────────────────────────
    #  TIMED MOVE (for fixed duration steps)
    # ─────────────────────────────────────────
    def _timed_move(self, linear, angular, duration):
        end = time.time() + duration
        while time.time() < end:
            if not self.navigating:
                self._stop()
                return False
            self._pub(linear, angular)
            time.sleep(0.05)
        self._stop()
        time.sleep(0.1)
        return True

    # ─────────────────────────────────────────
    #  NAVIGATION — odom waypoint following
    #  Moves to exact (x,y) using live odom
    # ─────────────────────────────────────────
    def _navigate_to_zone(self, dst):
        with self.nav_lock:
            if self.navigating:
                self.get_logger().warn('Already navigating — say STOP first')
                return
            self.navigating = True

        dst = dst.lower().strip()
        if dst not in ZONES:
            self.get_logger().warn(f'Unknown zone: {dst}')
            self.navigating = False
            return

        target_x, target_y = ZONES[dst]
        self.get_logger().info(
            f'=== Navigating to {dst.upper()} target=({target_x},{target_y}) '
            f'from ({self.x:.2f},{self.y:.2f}) ===')

        # Waypoints: first go to corridor (y=0, target_x),
        # then enter room (target_x, target_y)
        waypoints = [
            (self.x,    0.0),       # Step 1: align to corridor y=0
            (target_x,  0.0),       # Step 2: move along corridor to room x
            (target_x,  target_y),  # Step 3: enter room
        ]

        for wx, wy in waypoints:
            if not self.navigating:
                break
            if not self._move_to(wx, wy):
                break

        if self.navigating:
            self.current_zone = dst
            self.get_logger().info(f'=== Arrived at {dst.upper()} ===')
        self.navigating = False

    def _move_to(self, tx, ty):
        """Drive to (tx, ty) using live odometry feedback."""
        POSITION_TOLERANCE = 0.25  # metres
        MAX_TIME = 60.0            # safety timeout

        start = time.time()
        while self.navigating and (time.time() - start) < MAX_TIME:
            dx = tx - self.x
            dy = ty - self.y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist < POSITION_TOLERANCE:
                self._stop()
                time.sleep(0.1)
                return True

            # Desired heading
            desired_yaw = math.atan2(dy, dx)
            yaw_err = desired_yaw - self.yaw
            while yaw_err >  math.pi: yaw_err -= 2 * math.pi
            while yaw_err < -math.pi: yaw_err += 2 * math.pi

            # Wall check for forward motion — use tighter threshold during nav
            if self.front_dist < WALL_DIST_NAV and abs(yaw_err) < 0.3:
                self.get_logger().warn(
                    f'Wall detected ({self.front_dist:.2f}m) — stopping navigation')
                self._stop()
                self.navigating = False
                return False

            # P-controller: turn then drive
            angular = max(-TURN_RATE, min(TURN_RATE, 2.5 * yaw_err))
            # Only drive forward when roughly facing target
            linear = SPEED if abs(yaw_err) < 0.4 else 0.0

            self._pub(linear, angular)
            time.sleep(0.05)

        self._stop()
        if time.time() - start >= MAX_TIME:
            self.get_logger().warn('Navigation timeout')
        return False

    # ─────────────────────────────────────────
    #  NLU — Groq primary, local fallback
    # ─────────────────────────────────────────
    def _nlu(self, text):
        # Try Groq first
        if self.groq_available and self.groq_client:
            try:
                resp = self.groq_client.chat.completions.create(
                    model='llama-3.1-8b-instant',
                    messages=[{'role': 'system', 'content': NLU_PROMPT},
                               {'role': 'user',   'content': text}],
                    max_tokens=60, temperature=0.0)
                raw = re.sub(r'```json|```', '',
                             resp.choices[0].message.content.strip()).strip()
                result = json.loads(raw)
                self.get_logger().info(f'Groq NLU: {result}')
                return result
            except Exception as e:
                err = str(e)
                if '429' in err or 'quota' in err.lower() or 'rate' in err.lower():
                    self.get_logger().warn('Groq quota exceeded — switching to local NLU')
                    self.groq_available = False
                else:
                    self.get_logger().warn(f'Groq NLU error: {e}')

        # Local fallback
        result = local_nlu(text)
        self.get_logger().info(f'Local NLU: {result}')
        return result

    # ─────────────────────────────────────────
    #  DISPATCH
    # ─────────────────────────────────────────
    def _dispatch(self, text):
        t = text.lower().strip().rstrip('.')
        if len(t) < 2:
            return
        if t in NOISE or t.replace('.', '').strip() in NOISE:
            return

        self.get_logger().info(f'Heard: "{text}"')
        cmd = self._nlu(text)
        if not cmd:
            return

        action = cmd.get('action', 'unknown')

        if action == 'stop':
            self.navigating = False
            self.moving_continuous = False
            self._stop()
            self.get_logger().info('STOPPED')

        elif action == 'navigate':
            zone = cmd.get('zone', '')
            if zone:
                threading.Thread(
                    target=self._navigate_to_zone, args=(zone,), daemon=True).start()

        elif action == 'move_continuous':
            if self.navigating:
                self.get_logger().warn('Navigation in progress — say STOP first')
                return
            lin = float(cmd.get('linear', 0.0))
            ang = float(cmd.get('angular', 0.0))
            self._start_continuous(lin, ang)

        elif action == 'move':
            if self.navigating:
                self.get_logger().warn('Navigation in progress — say STOP first')
                return
            lin = float(cmd.get('linear', 0.0))
            ang = float(cmd.get('angular', 0.0))
            dur = float(cmd.get('duration', 1.0))
            threading.Thread(
                target=self._timed_move, args=(lin, ang, dur), daemon=True).start()

        elif action == 'unknown':
            self.get_logger().info('Command not recognised')

    # ─────────────────────────────────────────
    #  TRANSCRIPTION
    # ─────────────────────────────────────────
    def _transcribe(self, audio_np):
        # Groq Whisper (primary)
        if self.groq_client:
            try:
                import soundfile as sf
                buf = io.BytesIO()
                sf.write(buf, audio_np, SAMPLE_RATE, format='WAV')
                buf.seek(0); buf.name = 'audio.wav'
                result = self.groq_client.audio.transcriptions.create(
                    model='whisper-large-v3-turbo', file=buf, response_format='text')
                text = str(result).strip()
                if text:
                    return text
            except Exception as e:
                self.get_logger().warn(f'Groq Whisper: {e}')

        # faster-whisper (local fallback)
        if self.fw_model:
            try:
                segs, _ = self.fw_model.transcribe(
                    audio_np.astype(np.float32), language='en')
                text = ' '.join(s.text for s in segs).strip()
                if text:
                    return text
            except Exception as e:
                self.get_logger().warn(f'faster-whisper: {e}')
        return ''

    # ─────────────────────────────────────────
    #  VAD
    # ─────────────────────────────────────────
    def _has_speech(self, audio_np):
        if self.vad_model is None:
            return True
        try:
            import torch
            conf = self.vad_model(
                torch.from_numpy(audio_np.astype(np.float32)), SAMPLE_RATE).item()
            self.get_logger().info(f'VAD: {conf*100:.0f}%')
            return conf > 0.6
        except:
            return True

    # ─────────────────────────────────────────
    #  LISTEN LOOP
    # ─────────────────────────────────────────
    def _listen_loop(self):
        # Device 0 (pulse) confirmed working under WSLg
        device_index = 0
        try:
            devs = sd.query_devices()
            for i, d in enumerate(devs):
                if 'pulse' in d['name'].lower() and d['max_input_channels'] > 0:
                    device_index = i
                    break
            self.get_logger().info(f'Audio device {device_index}: {sd.query_devices(device_index)["name"]}')
        except Exception as e:
            self.get_logger().warn(f'Audio device query failed: {e}, using device 0')

        while rclpy.ok():
            try:
                audio = sd.rec(CHUNK_SAMPLES, samplerate=SAMPLE_RATE,
                               channels=1, dtype='float32', device=device_index)
                sd.wait()
                audio_np = np.clip(audio.flatten() * 3.0, -1.0, 1.0)
                if not self._has_speech(audio_np):
                    continue
                text = self._transcribe(audio_np)
                if not text or len(text) < 2:
                    continue
                self._dispatch(text)
            except Exception as e:
                self.get_logger().error(f'Listen error: {e}')
                time.sleep(1.0)


def main(args=None):
    rclpy.init(args=args)
    node = VoiceControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
