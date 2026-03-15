#!/usr/bin/env python3
import os
os.environ.setdefault('PULSE_SERVER', 'unix:/mnt/wslg/PulseServer')

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import threading
import time
import math
import queue
import re
import numpy as np
import sounddevice as sd
import io, json

# ─────────────────────────────────────────────
#  WORLD LAYOUT
#  NW: RECEPTION  (-6, +6)  — orange
#  NE: ICU        (+6, +6)  — yellow
#  SW: WARD       (-6, -6)  — blue
#  SE: PHARMACY   (+6, -6)  — green
#  Corridor at y=0, walls at y=±2, gaps at x=±6
#  Boundary walls at x=±11, y=±11
# ─────────────────────────────────────────────

SPEED         = 0.5
TURN_RATE     = 1.5
WALL_DIST     = 0.4
WALL_DIST_NAV = 0.15

SAMPLE_RATE   = 16000
CHUNK_SAMPLES = int(SAMPLE_RATE * 1.5)

T90  = round(math.pi / 2 / TURN_RATE, 3)
T180 = round(math.pi     / TURN_RATE, 3)
T360 = round(2 * math.pi / TURN_RATE, 3)

ZONES = {
    'icu':       ( 6.0,  6.0),
    'pharmacy':  ( 6.0, -6.0),
    'reception': (-6.0,  6.0),
    'ward':      (-6.0, -6.0),
    'center':    ( 0.0,  0.0),
}

CARDINAL_YAW = {
    'east':       0.0,         'west':       math.pi,
    'north':      math.pi/2,   'south':     -math.pi/2,
    'northeast':  math.pi/4,   'northwest':  3*math.pi/4,
    'southeast': -math.pi/4,   'southwest': -3*math.pi/4,
}

# ─────────────────────────────────────────────
#  GROQ NLU PROMPT
# ─────────────────────────────────────────────
NLU_SYSTEM = f"""You are the AI brain of a hospital delivery robot. Understand voice commands even with:
- Background noise
- Accents or mispronunciation
- Garbled speech recognition output

Use semantic understanding to figure out INTENT, not just keywords.

Hospital zones:
- ICU (yellow, northeast) — "intensive care", "ICU", "yellow", "i see you", "aicu"
- Pharmacy (green, southeast) — "pharmacy", "farmasi", "medicine", "green", "dispensary"
- Reception (orange, northwest) — "reception", "front desk", "lobby", "orange", "auto reception"
- Ward (blue, southwest) — "ward", "patient room", "blue"
- Center — "center", "centre", "middle", "home", "reset", "origin"

Return ONLY JSON. No markdown, no explanation.

NAVIGATION:
{{"action":"navigate","zone":"icu"}}
{{"action":"navigate","zone":"pharmacy"}}
{{"action":"navigate","zone":"reception"}}
{{"action":"navigate","zone":"ward"}}
{{"action":"navigate","zone":"center"}}

FIXED TURNS:
- right 90°: {{"action":"move","linear":0.0,"angular":{-TURN_RATE},"duration":{T90}}}
- left 90°:  {{"action":"move","linear":0.0,"angular":{TURN_RATE},"duration":{T90}}}
- 180°/turn around/donor: {{"action":"move","linear":0.0,"angular":{TURN_RATE},"duration":{T180}}}
- spin/360°: {{"action":"move","linear":0.0,"angular":{TURN_RATE},"duration":{T360}}}

CONTINUOUS (runs until stop):
- forward/go/ahead: {{"action":"move_continuous","linear":{SPEED},"angular":0.0}}
- back/reverse:     {{"action":"move_continuous","linear":{-SPEED},"angular":0.0}}

DISTANCE:
- "go forward 3 meters": {{"action":"move","linear":{SPEED},"angular":0.0,"duration":<m/{SPEED}>}}
- "go back 2 meters":    {{"action":"move","linear":{-SPEED},"angular":0.0,"duration":<m/{SPEED}>}}

CARDINAL:
- "go east":          {{"action":"cardinal","direction":"east","duration":null}}
- "go east 3 meters": {{"action":"cardinal","direction":"east","duration":{round(3/SPEED,2)}}}
- Supports: east, west, north, south, northeast, northwest, southeast, southwest

STOP:
- stop/halt/freeze/wait/stopp: {{"action":"stop"}}

COMPOUND (multiple commands → array):
"turn right and go forward" → [{{"action":"move","linear":0.0,"angular":{-TURN_RATE},"duration":{T90}}},{{"action":"move_continuous","linear":{SPEED},"angular":0.0}}]

NOISE: garbled words like "donor"=turn around, "take lift"=turn left, "farmasi"=pharmacy, "stopp"=stop, "tyk raut"=turn right
Unrecognisable noise → {{"action":"unknown"}}"""

# ─────────────────────────────────────────────
#  LOCAL FALLBACK NLU
# ─────────────────────────────────────────────
FALLBACK_PATTERNS = [
    (r'\b(stop|halt|freeze|cancel|pause|wait|enough|brake|stopp)\b',
     {'action': 'stop'}),
    (r'\b(icu|i\.c\.u|intensive care|yellow|aicu)\b',
     {'action': 'navigate', 'zone': 'icu'}),
    (r'pharmac|pharma|farmasi|\bgreen\b|\bmedicine\b|\bdispensary\b',
     {'action': 'navigate', 'zone': 'pharmacy'}),
    (r'\b(reception|front desk|lobby|orange|receptionist|recep)\b',
     {'action': 'navigate', 'zone': 'reception'}),
    (r'\bward\b|\bblue room\b|\bpatient ward\b',
     {'action': 'navigate', 'zone': 'ward'}),
    (r'\b(center|centre|middle|home|reset|origin)\b',
     {'action': 'navigate', 'zone': 'center'}),
    (r'\b(turn around|u.turn|180|donor|turnaround)\b',
     {'action': 'move', 'linear': 0.0, 'angular': TURN_RATE, 'duration': T180}),
    (r'\b(spin|360)\b',
     {'action': 'move', 'linear': 0.0, 'angular': TURN_RATE, 'duration': T360}),
    (r'\b(turn left|take left|go left|rotate left|take lift|tick left|left turn)\b',
     {'action': 'move', 'linear': 0.0, 'angular': TURN_RATE, 'duration': T90}),
    (r'\b(turn right|take right|go right|rotate right|right turn)\b',
     {'action': 'move', 'linear': 0.0, 'angular': -TURN_RATE, 'duration': T90}),
    (r'\b(go|head|move)\s+east\b',      {'action': 'cardinal', 'direction': 'east',      'duration': None}),
    (r'\b(go|head|move)\s+west\b',      {'action': 'cardinal', 'direction': 'west',      'duration': None}),
    (r'\b(go|head|move)\s+north\b',     {'action': 'cardinal', 'direction': 'north',     'duration': None}),
    (r'\b(go|head|move)\s+south\b',     {'action': 'cardinal', 'direction': 'south',     'duration': None}),
    (r'\b(go|head|move)\s+northeast\b', {'action': 'cardinal', 'direction': 'northeast', 'duration': None}),
    (r'\b(go|head|move)\s+northwest\b', {'action': 'cardinal', 'direction': 'northwest', 'duration': None}),
    (r'\b(go|head|move)\s+southeast\b', {'action': 'cardinal', 'direction': 'southeast', 'duration': None}),
    (r'\b(go|head|move)\s+southwest\b', {'action': 'cardinal', 'direction': 'southwest', 'duration': None}),
    (r'\b(forward|go forward|move forward|go ahead|advance|ahead|straight)\b',
     {'action': 'move_continuous', 'linear': SPEED, 'angular': 0.0}),
    (r'\b(back|backward|go back|reverse|retreat)\b',
     {'action': 'move_continuous', 'linear': -SPEED, 'angular': 0.0}),
]

def local_nlu(text):
    t = text.lower().strip()
    # Distance movement
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:meter|metre|m)\b', t)
    if m:
        dist = float(m.group(1))
        if re.search(r'\b(back|backward|reverse)\b', t):
            return {'action': 'move', 'linear': -SPEED, 'angular': 0.0, 'duration': round(dist/SPEED, 2)}
        if re.search(r'\b(forward|ahead)\b', t):
            return {'action': 'move', 'linear': SPEED, 'angular': 0.0, 'duration': round(dist/SPEED, 2)}
    for pattern, result in FALLBACK_PATTERNS:
        if re.search(pattern, t):
            return result
    return {'action': 'unknown'}

NOISE_WORDS = {'gracias','merci','danke','okay','ok','thank you','thanks','yes','no',
               'hmm','uh','um','ah','bye','hello','hi','yeah','sure','the','a','an'}


class VoiceControlNode(Node):
    def __init__(self):
        super().__init__('voice_control')

        self.pub      = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

        self.x = 0.0; self.y = 0.0; self.yaw = 0.0
        self.front_dist = 999.0
        self.navigating = False
        self.moving_continuous = False
        self.moving_timed = False
        self.current_zone = None
        self.nav_lock = threading.Lock()

        # Groq
        self.groq_client = None
        self.groq_available = False
        try:
            from groq import Groq
            self.groq_client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
            self.groq_available = True
            self.get_logger().info('Groq AI ready')
        except Exception as e:
            self.get_logger().warn(f'Groq unavailable: {e} — using local NLU')

        # faster-whisper
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

        self.get_logger().info('=== AI Voice Control Ready ===')
        self.get_logger().info('=== Commands: go to ICU/pharmacy/ward/reception, forward, back, turn left/right, stop ===')
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def _odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0*(q.w*q.z + q.x*q.y)
        cosy = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
        self.yaw = math.atan2(siny, cosy)

    def _scan_cb(self, msg):
        ranges = msg.ranges
        if not ranges: return
        n = len(ranges)
        idxs = list(range(0, int(n*15/360)+1)) + list(range(int(n*345/360), n))
        valid = [ranges[i] for i in idxs
                 if not math.isinf(ranges[i]) and not math.isnan(ranges[i]) and ranges[i] > 0.05]
        self.front_dist = min(valid) if valid else 999.0

    def _pub(self, linear, angular):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.pub.publish(msg)

    def _stop(self):
        self._pub(0.0, 0.0)

    def _stop_firm(self):
        for _ in range(10):
            self._stop()
            time.sleep(0.05)

    # ── Continuous movement ──
    def _start_continuous(self, linear, angular):
        def _run():
            self.moving_continuous = True
            self.get_logger().info(f'Moving: lin={linear:.1f} ang={angular:.1f} — say STOP')
            while self.moving_continuous:
                if linear > 0 and self.front_dist < WALL_DIST:
                    self.get_logger().warn(f'Wall at {self.front_dist:.2f}m — stopping')
                    break
                self._pub(linear, angular)
                time.sleep(0.05)
            self._stop_firm()
            self.moving_continuous = False
            self.get_logger().info('Stopped')
        threading.Thread(target=_run, daemon=True).start()

    # ── Timed move ──
    def _timed_move(self, linear, angular, duration):
        end = time.time() + duration
        while time.time() < end:
            self._pub(linear, angular)
            time.sleep(0.05)
        self._stop()
        time.sleep(0.1)

    # ── Cardinal direction ──
    def _cardinal_move(self, direction, duration=None):
        target_yaw = CARDINAL_YAW.get(direction, 0.0)
        yaw_err = target_yaw - self.yaw
        while yaw_err >  math.pi: yaw_err -= 2*math.pi
        while yaw_err < -math.pi: yaw_err += 2*math.pi
        if abs(yaw_err) > 0.1:
            self.moving_timed = True
            self._timed_move(0.0, TURN_RATE if yaw_err > 0 else -TURN_RATE, abs(yaw_err)/TURN_RATE)
            self.moving_timed = False
        if duration:
            self.moving_timed = True
            self._timed_move(SPEED, 0.0, duration)
            self.moving_timed = False
        else:
            self._start_continuous(SPEED, 0.0)

    # ── Navigation — 4-step room-exit routing ──
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
            f'=== Navigating to {dst.upper()} ({target_x},{target_y}) '
            f'from ({self.x:.2f},{self.y:.2f}) ===')

        if dst == 'center':
            waypoints = [(0.0, 0.0)]
        else:
            waypoints = []
            # Determine which gap to use based on target
            gap_x = target_x  # use target's x gap (east=+6, west=-6)

            # Step 1: If inside a room (|y| > 2.2), must exit through correct gap
            if abs(self.y) > 2.2:
                # Step 1a: Move to gap x at current y (align horizontally first)
                if abs(self.x - gap_x) > 0.5:
                    waypoints.append((gap_x, self.y))
                # Step 1b: Exit through gap to just inside corridor
                exit_y = 1.5 if self.y > 0 else -1.5
                waypoints.append((gap_x, exit_y))
            # Step 2: Get to corridor center y=0
            waypoints.append((gap_x, 0.0))
            # Step 3: Travel along corridor if needed (already at gap_x)
            # Step 4: Enter target room
            waypoints.append((target_x, target_y))

        for wx, wy in waypoints:
            if not self.navigating: break
            self.get_logger().info(f'  Waypoint → ({wx:.1f},{wy:.1f})')
            self._move_to(wx, wy)

        if self.navigating:
            dx = target_x - self.x
            dy = target_y - self.y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 1.5:
                self.current_zone = dst
                self._stop_firm()
                self.get_logger().info(f'=== Arrived at {dst.upper()} ===')
            else:
                self._stop_firm()
                self.get_logger().warn(f'Did not reach {dst.upper()} ({dist:.1f}m away) — say command again')
        self.navigating = False

    def _move_to(self, tx, ty):
        TOLERANCE = 0.3
        MAX_TIME  = 30.0
        start = time.time()
        wall_hits = 0

        while self.navigating and (time.time() - start) < MAX_TIME:
            dx = tx - self.x; dy = ty - self.y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < TOLERANCE:
                self._stop_firm()
                return

            desired_yaw = math.atan2(dy, dx)
            yaw_err = desired_yaw - self.yaw
            while yaw_err >  math.pi: yaw_err -= 2*math.pi
            while yaw_err < -math.pi: yaw_err += 2*math.pi

            # Wall recovery
            if self.front_dist < WALL_DIST_NAV and abs(yaw_err) < 0.4:
                wall_hits += 1
                self.get_logger().warn(f'Wall hit #{wall_hits} — backing up')
                end = time.time() + 0.8
                while time.time() < end and self.navigating:
                    self._pub(-SPEED * 0.5, 0.0)
                    time.sleep(0.05)
                self._stop()
                time.sleep(0.2)
                if wall_hits > 3:
                    self.get_logger().warn('Skipping waypoint')
                    return
                continue

            angular = max(-TURN_RATE, min(TURN_RATE, 3.0 * yaw_err))
            speed   = SPEED if dist > 1.0 else SPEED * 0.4
            linear  = speed if abs(yaw_err) < 0.3 else 0.0
            self._pub(linear, angular)
            time.sleep(0.05)

        self._stop()
        self.get_logger().warn(f'Waypoint ({tx:.1f},{ty:.1f}) timeout')

    # ── NLU — Local first, Groq for unclear commands ──
    def _nlu(self, text):
        # Step 1: Try local NLU — zero API calls for standard commands
        local_result = local_nlu(text)
        if local_result.get('action') != 'unknown':
            self.get_logger().info(f'Local NLU: {local_result}')
            return local_result

        # Step 2: Groq for accented/garbled/complex speech
        if self.groq_available and self.groq_client:
            try:
                resp = self.groq_client.chat.completions.create(
                    model='llama-3.3-70b-versatile',
                    messages=[
                        {'role': 'system', 'content': NLU_SYSTEM},
                        {'role': 'user',   'content': f'Transcription: "{text}"'}
                    ],
                    max_tokens=150, temperature=0.0)
                raw = re.sub(r'```json|```', '',
                             resp.choices[0].message.content.strip()).strip()
                result = json.loads(raw)
                self.get_logger().info(f'Groq NLU: {result}')
                return result
            except Exception as e:
                err = str(e)
                if '429' in err or 'quota' in err.lower() or 'rate' in err.lower():
                    self.get_logger().warn('Groq quota hit — local NLU only (restores in 60s)')
                    self.groq_available = False
                    def restore():
                        time.sleep(60.0)
                        self.groq_available = True
                        self.get_logger().info('Groq restored')
                    threading.Thread(target=restore, daemon=True).start()
                else:
                    self.get_logger().warn(f'Groq error: {e}')

        return local_result

    # ── Dispatch ──
    def _dispatch(self, text):
        t = text.lower().strip().rstrip('.')
        if len(t) < 2: return
        if t in NOISE_WORDS: return

        self.get_logger().info(f'Heard: "{text}"')
        result = self._nlu(text)
        if not result: return

        commands = result if isinstance(result, list) else [result]
        for cmd in commands:
            action = cmd.get('action', 'unknown')

            if action == 'stop':
                self.navigating = False
                self.moving_continuous = False
                self.moving_timed = False
                self._stop_firm()
                self.get_logger().info('STOPPED')
                return

            elif action == 'navigate':
                zone = cmd.get('zone', '')
                if zone:
                    threading.Thread(
                        target=self._navigate_to_zone, args=(zone,), daemon=True).start()

            elif action == 'move_continuous':
                if self.navigating:
                    self.get_logger().warn('Navigating — say STOP first'); return
                if self.moving_timed:
                    self.get_logger().warn('Move in progress — wait'); return
                self.moving_continuous = False
                time.sleep(0.1)
                self._start_continuous(
                    float(cmd.get('linear', 0.0)),
                    float(cmd.get('angular', 0.0)))

            elif action == 'move':
                if self.navigating:
                    self.get_logger().warn('Navigating — say STOP first'); return
                if self.moving_timed:
                    self.get_logger().warn('Move in progress — wait'); return
                lin = float(cmd.get('linear', 0.0))
                ang = float(cmd.get('angular', 0.0))
                dur = float(cmd.get('duration', 1.0))
                def _run(l=lin, a=ang, d=dur):
                    self.moving_timed = True
                    self.get_logger().info(f'Timed move: lin={l:.1f} ang={a:.1f} dur={d:.2f}s')
                    self._timed_move(l, a, d)
                    time.sleep(0.2)
                    self.moving_timed = False
                    self.get_logger().info('Move complete — ready')
                threading.Thread(target=_run, daemon=True).start()

            elif action == 'cardinal':
                if self.navigating:
                    self.get_logger().warn('Navigating — say STOP first'); return
                threading.Thread(
                    target=self._cardinal_move,
                    args=(cmd.get('direction', 'east'), cmd.get('duration')),
                    daemon=True).start()

            elif action == 'unknown':
                self.get_logger().info('Could not understand command')

    # ── Transcription — Groq Whisper primary, faster-whisper fallback ──
    def _transcribe(self, audio_np):
        if self.groq_client:
            try:
                import soundfile as sf
                buf = io.BytesIO()
                sf.write(buf, audio_np, SAMPLE_RATE, format='WAV')
                buf.seek(0); buf.name = 'audio.wav'
                result = self.groq_client.audio.transcriptions.create(
                    model='whisper-large-v3-turbo', file=buf,
                    response_format='text', language='en',
                    prompt='Hospital robot commands: go to pharmacy, go to ICU, go to ward, go to reception, turn left, turn right, go forward, go back, stop, turn around, spin.')
                text = str(result).strip()
                if text: return text
            except Exception as e:
                self.get_logger().warn(f'Whisper: {e}')
        if self.fw_model:
            try:
                segs, _ = self.fw_model.transcribe(audio_np.astype(np.float32), language='en')
                text = ' '.join(s.text for s in segs).strip()
                if text: return text
            except: pass
        return ''

    # ── VAD — energy + silero ──
    def _has_speech(self, audio_np):
        energy = np.sqrt(np.mean(audio_np ** 2))
        if energy < 0.02: return False
        if self.vad_model is None: return True
        try:
            import torch
            conf = self.vad_model(
                torch.from_numpy(audio_np.astype(np.float32)), SAMPLE_RATE).item()
            return conf > 0.5
        except:
            return True

    # ── Listen loop — continuous callback stream ──
    def _listen_loop(self):
        audio_queue = queue.Queue(maxsize=5)
        buffer = []
        chunks_processed = 0

        def get_device():
            try:
                devs = sd.query_devices()
                for i, d in enumerate(devs):
                    if 'pulse' in d['name'].lower() and d['max_input_channels'] > 0:
                        self.get_logger().info(f'Audio: {d["name"]}')
                        return i
            except: pass
            return None

        def audio_callback(indata, frames, time_info, status):
            buffer.append(indata.copy())
            total = sum(len(b) for b in buffer)
            if total >= CHUNK_SAMPLES:
                chunk = np.concatenate(buffer)[:CHUNK_SAMPLES].flatten()
                buffer.clear()
                try:
                    audio_queue.put_nowait(chunk)
                except: pass

        def process_loop():
            nonlocal chunks_processed
            self.get_logger().info('Process thread started')
            while rclpy.ok():
                try:
                    audio_np = audio_queue.get(timeout=1.0)
                    chunks_processed += 1
                    audio_np = np.clip(audio_np * 3.0, -1.0, 1.0)
                    if not self._has_speech(audio_np): continue
                    text = self._transcribe(audio_np)
                    if not text or len(text) < 2: continue
                    self._dispatch(text)
                except queue.Empty:
                    continue
                except Exception as e:
                    self.get_logger().error(f'Process error: {e}')
                    time.sleep(0.1)

        def heartbeat():
            while rclpy.ok():
                time.sleep(15.0)
                self.get_logger().info(f'Listening — chunks: {chunks_processed}')
        threading.Thread(target=heartbeat, daemon=True).start()
        threading.Thread(target=process_loop, daemon=True).start()

        device_index = get_device()
        while rclpy.ok():
            try:
                with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                    dtype='float32', device=device_index,
                                    blocksize=1024, callback=audio_callback):
                    self.get_logger().info('Microphone active — speak your command')
                    while rclpy.ok():
                        time.sleep(0.5)
            except Exception as e:
                self.get_logger().error(f'Mic error: {e} — restarting')
                buffer.clear()
                time.sleep(1.0)
                device_index = get_device()


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
