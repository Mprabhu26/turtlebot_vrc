#!/usr/bin/env python3
import os
os.environ['PULSE_SERVER'] = '/mnt/wslg/runtime-dir/pulse/native'

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory

import sounddevice as sd
import numpy as np
import json
import threading
import time
import re
import queue
import datetime
import tempfile
import subprocess

import soundfile as sf
from scipy import signal as scipy_signal
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, get_speech_timestamps
import torch
from groq import Groq
from fuzzywuzzy import fuzz
import yaml



# ── Hospital Zone Coordinates (x, y, yaw) ──────────────────────────────
HOSPITAL_ZONES = {
    'reception':  (-8.0, -3.0, 0.0),
    'pharmacy':   (-4.0, -3.0, 0.0),
    'icu':        ( 7.0, -3.0, 0.0),
    'corridor':   ( 0.0, -3.0, 0.0),
    'patient room':( 0.0, -4.5, 0.0),
    'ward':       ( 0.0, -4.5, 0.0),
}

# ─────────────────────────────────────────────
#  Quota Tracker
# ─────────────────────────────────────────────
class QuotaTracker:
    DAILY_LIMIT = 1400

    def __init__(self):
        self.count             = 0
        self.date              = datetime.date.today()
        self.last_call_time    = 0
        self.permanently_blocked = False

    def can_use(self):
        if datetime.date.today() != self.date:
            self.count = 0
            self.date  = datetime.date.today()
            self.permanently_blocked = False
        if self.permanently_blocked:
            return False
        if time.time() - self.last_call_time < 5.0:
            return False
        return self.count < self.DAILY_LIMIT

    def increment(self):
        self.count += 1
        self.last_call_time = time.time()

    @property
    def remaining(self):
        return max(0, self.DAILY_LIMIT - self.count)


# ─────────────────────────────────────────────
#  Rule-based NLU (fallback)
# ─────────────────────────────────────────────
class RuleNLU:
    WORD_NUMBERS = {
        'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,
        'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
        'eleven':11,'twelve':12,'fifteen':15,'twenty':20,
        'thirty':30,'forty':40,'fifty':50,'hundred':100,
        'half':0.5,'quarter':0.25,'a':1,'an':1,
    }
    COMPASS = {
        'northeast':( 0.2,-0.3),'northwest':( 0.2, 0.3),
        'southeast':(-0.2,-0.3),'southwest':(-0.2, 0.3),
        'north'    :( 0.2, 0.0),'south'    :(-0.2, 0.0),
        'east'     :( 0.0,-0.5),'west'     :( 0.0, 0.5),
    }

    def normalize(self, text):
        t = text.lower().strip()
        t = re.sub(r'\s+', ' ', t)
        t = re.sub(r'\bne\b','northeast',t)
        t = re.sub(r'\bnw\b','northwest',t)
        t = re.sub(r'\bse\b','southeast',t)
        t = re.sub(r'\bsw\b','southwest',t)
        t = t.replace('north east','northeast').replace('north west','northwest')
        t = t.replace('south east','southeast').replace('south west','southwest')
        for w, n in self.WORD_NUMBERS.items():
            t = re.sub(rf'\b{w}\b', str(n), t)
        return t

    def extract_distance(self, t):
        for pat, mult in [
            (r'(\d+\.?\d*)\s*(?:km|kilometers?)', 1000.0),
            (r'(\d+\.?\d*)\s*(?:meters?|metres?|m\b)', 1.0),
            (r'(\d+\.?\d*)\s*(?:cm|centimeters?)', 0.01),
        ]:
            m = re.search(pat, t)
            if m: return float(m.group(1)) * mult
        return None

    def extract_duration(self, t):
        for pat, mult in [
            (r'(\d+\.?\d*)\s*(?:minutes?|mins?)', 60.0),
            (r'(\d+\.?\d*)\s*(?:seconds?|secs?|s\b)', 1.0),
        ]:
            m = re.search(pat, t)
            if m: return float(m.group(1)) * mult
        return None

    def parse(self, raw):
        t     = self.normalize(raw)
        speed = 0.5 if any(w in t for w in ['slow','slowly','creep']) else \
                2.0 if any(w in t for w in ['fast','quick','hurry']) else 1.0
        dist  = self.extract_distance(t)
        dur   = self.extract_duration(t)
        lx    = 0.2 * speed

        if any(w in t for w in ['stop','halt','freeze','pause']):
            return dict(linear_x=0.0, angular_z=0.0, duration=None, explanation='Stop')

        if any(w in t for w in ['spin','rotate','turn around']):
            count = 1
            m = re.search(r'(\d+\.?\d*)\s*(?:times?|rotations?)', t)
            if m: count = float(m.group(1))
            return dict(linear_x=0.0, angular_z=1.0,
                        duration=count*6.28, explanation=f'Spin {count}x')

        for direction, (dlx, daz) in self.COMPASS.items():
            if direction in t:
                dlx *= speed
                d2 = dist/abs(dlx) if dist and dlx!=0 else dur
                return dict(linear_x=dlx, angular_z=daz, duration=d2,
                            explanation=f'Go {direction}')

        if any(w in t for w in ['left','turn left','rotate left']):
            return dict(linear_x=0.0, angular_z=0.5, duration=dur, explanation='Turn left')
        if any(w in t for w in ['right','turn right','rotate right']):
            return dict(linear_x=0.0, angular_z=-0.5, duration=dur, explanation='Turn right')
        if any(w in t for w in ['back','backward','reverse','retreat']):
            d2 = dist/lx if dist else dur
            return dict(linear_x=-lx, angular_z=0.0, duration=d2, explanation='Backward')
        if any(w in t for w in ['forward','ahead','straight','advance']) or dist or dur:
            d2 = dist/lx if dist else dur
            return dict(linear_x=lx, angular_z=0.0, duration=d2, explanation='Forward')

        return None


# ─────────────────────────────────────────────
#  Groq NLU
# ─────────────────────────────────────────────
class GroqNLU:
    SYSTEM = """You are a robot command interpreter for TurtleBot3.
The user speaks commands. The speech may have accent or slight mishearing.
Extract the robot movement intent and return ONLY a JSON object.

Format:
{
  "action": "move|stop|unknown",
  "linear_x": <float m/s, +forward/-backward>,
  "angular_z": <float rad/s, +left/-right>,
  "duration": <float seconds or null>,
  "explanation": "<brief>"
}

Rules:
- stop/halt/freeze → 0,0
- forward/north/ahead → linear_x=0.2
- backward/south/back → linear_x=-0.2
- left/west → angular_z=0.5
- right/east → angular_z=-0.5
- northeast → linear_x=0.2, angular_z=-0.3
- northwest → linear_x=0.2, angular_z=0.3
- southeast → linear_x=-0.2, angular_z=-0.3
- southwest → linear_x=-0.2, angular_z=0.3
- slow → multiply by 0.5, fast → multiply by 2.0
- spin → angular_z=1.0
- distance meters: duration = meters/0.2
- seconds/minutes: use as duration
- word numbers: one=1, two=2, three=3, half=0.5
- noise/music/unclear → action=unknown
"""

    def __init__(self, client):
        self.client = client

    def parse(self, text):
        try:
            resp = self.client.chat.completions.create(
                model='llama-3.1-8b-instant',
                messages=[
                    {'role':'system', 'content': self.SYSTEM},
                    {'role':'user',   'content': f'Command: "{text}"'},
                ],
                temperature=0.1,
                max_tokens=150,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r'```(?:json)?', '', raw).strip()
            cmd = json.loads(raw)
            return None if cmd.get('action') == 'unknown' else cmd
        except Exception as e:
            raise RuntimeError(f'Groq NLU error: {e}')


# ─────────────────────────────────────────────
#  ROS 2 Node
# ─────────────────────────────────────────────
class VoiceControl(Node):

    def __init__(self):
        super().__init__('voice_control')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.audio_q = queue.Queue()
        self.audio_buffer   = []
        self.silence_frames = 0
        self.speaking       = False
        self.quota          = QuotaTracker()
        self.rule_nlu       = RuleNLU()

        # Groq client
        api_key = os.environ.get('GROQ_API_KEY', '')
        if api_key:
            self.groq_client  = Groq(api_key=api_key)
            self.groq_nlu     = GroqNLU(self.groq_client)
            self.groq_whisper = True
            self.get_logger().info('Groq ready (NLU + Whisper)')
        else:
            self.groq_client  = None
            self.groq_nlu     = None
            self.groq_whisper = False
            self.get_logger().warn('No GROQ_API_KEY — rule NLU + faster-whisper only')

        # faster-whisper (offline fallback)
        self.get_logger().info('Loading faster-whisper (base)...')
        self.fw_model = WhisperModel('base', device='cpu', compute_type='int8')
        self.get_logger().info('faster-whisper ready')

        # Silero VAD
        self.vad_model = load_silero_vad()
        self.get_logger().info('Silero VAD ready')

        # Audio stream
        self._start_audio()

        # Processing thread
        threading.Thread(target=self.process_loop, daemon=True).start()
        self.get_logger().info('Voice Control Node ready — speak a command!')

    # ── Audio ──────────────────────────────────────────────────────────
    def _start_audio(self):
        for attempt in range(3):
            try:
                if attempt > 0:
                    self.get_logger().info(f'Restarting PulseAudio (attempt {attempt+1})...')
                    subprocess.run(['pulseaudio','--kill'], capture_output=True)
                    time.sleep(2)
                    subprocess.run(['pulseaudio','--start'], capture_output=True)
                    time.sleep(2)
                    sd._terminate(); sd._initialize()

                devices = sd.query_devices()
                dev = next((i for i,d in enumerate(devices)
                            if d['max_input_channels']>0
                            and 'pulse' in d['name'].lower()), None)
                self.get_logger().info(f'Audio device: {dev}')
                self.stream = sd.InputStream(
                    samplerate=16000, channels=1, blocksize=2000,
                    device=dev, dtype='float32',
                    callback=self.audio_callback
                )
                self.stream.start()
                self.get_logger().info('Microphone open!')
                return
            except Exception as e:
                self.get_logger().warn(f'Audio attempt {attempt+1} failed: {e}')
                time.sleep(1)
        self.get_logger().error('Audio failed — check PulseAudio')
        rclpy.shutdown()

    def audio_callback(self, indata, frames, time_info, status):
        volume = float(np.abs(indata).mean())
        is_speech = volume > 0.008

        if is_speech:
            self.audio_buffer.append(indata.copy())
            self.silence_frames = 0
            self.speaking = True
        elif self.speaking:
            self.audio_buffer.append(indata.copy())
            self.silence_frames += 1
            if self.silence_frames >= 16:  # ~2s silence = end of utterance
                audio = np.concatenate(self.audio_buffer).flatten()
                if len(audio) > 8000:  # min 0.5s
                    self.audio_q.put(audio)
                self.audio_buffer   = []
                self.silence_frames = 0
                self.speaking       = False

    # ── Audio enhancement ──────────────────────────────────────────────
    def enhance(self, audio):
        audio = audio * 3.0
        floor = np.percentile(np.abs(audio), 20) * 2.5
        audio = np.where(np.abs(audio) < floor, audio * 0.1, audio)
        sos   = scipy_signal.butter(4, 80, btype='high', fs=16000, output='sos')
        audio = scipy_signal.sosfilt(sos, audio)
        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    # ── Main processing loop ───────────────────────────────────────────
    def process_loop(self):
        while rclpy.ok():
            try:
                audio = self.audio_q.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                # Enhance audio
                audio = self.enhance(audio)

                # VAD check
                tensor = torch.FloatTensor(audio)
                timestamps = get_speech_timestamps(
                    tensor, self.vad_model,
                    sampling_rate=16000,
                    threshold=0.6,
                    min_speech_duration_ms=300,
                )
                if not timestamps:
                    self.get_logger().debug('VAD: no speech — skipped')
                    continue

                speech = np.concatenate([
                    audio[t['start']:t['end']] for t in timestamps
                ])
                ratio = len(speech)/len(audio)
                self.get_logger().info(f'VAD: speech {ratio*100:.0f}%')

                # Transcribe
                text = self._transcribe(speech)
                if not text:
                    continue

                self.get_logger().info(f'Transcribed: "{text}"')
                self.dispatch(text)

            except Exception as e:
                self.get_logger().error(f'Processing error: {e}')

    # ── Transcription ──────────────────────────────────────────────────
    def _transcribe(self, audio):
        # Try Groq Whisper first
        if self.groq_whisper and self.groq_client:
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    tmp = f.name
                sf.write(tmp, audio, 16000)
                with open(tmp, 'rb') as f:
                    resp = self.groq_client.audio.transcriptions.create(
                        file=(tmp, f.read()),
                        model='whisper-large-v3-turbo',
                        language='en',
                        response_format='text',
                    )
                os.unlink(tmp)
                text = resp.strip() if isinstance(resp, str) else resp.text.strip()
                if text:
                    self.get_logger().info(f'Groq Whisper: "{text}"')
                    return text
            except Exception as e:
                self.get_logger().warn(f'Groq Whisper failed: {e} — using faster-whisper')

        # faster-whisper fallback
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                tmp = f.name
            sf.write(tmp, audio, 16000)
            segs, _ = self.fw_model.transcribe(tmp, language='en', beam_size=1, vad_filter=True)
            os.unlink(tmp)
            text = ' '.join(s.text for s in segs).strip()
            if text:
                self.get_logger().info(f'faster-whisper: "{text}"')
            return text
        except Exception as e:
            self.get_logger().error(f'faster-whisper error: {e}')
            return None

    # ── NLU dispatch ───────────────────────────────────────────────────
    def dispatch(self, text):
        cmd = None

        # Try Groq NLU
        if self.groq_nlu and self.quota.can_use() and not self.quota.permanently_blocked:
            try:
                cmd = self.groq_nlu.parse(text)
                self.quota.increment()
                self.get_logger().info(f'Groq NLU ({self.quota.remaining} left): {cmd}')
            except Exception as e:
                err = str(e)
                if 'limit: 0' in err:
                    self.quota.permanently_blocked = True
                    self.get_logger().warn('Groq quota blocked — switching to rule NLU')
                else:
                    self.get_logger().warn(f'Groq NLU failed: {e}')
                cmd = None

        # Rule NLU fallback
        if cmd is None:
            cmd = self.rule_nlu.parse(text)
            if cmd:
                self.get_logger().info(f'Rule NLU: {cmd["explanation"]}')

        if cmd is None:
            self.get_logger().warn(f'Not understood: "{text}"')
            return

        # Check for zone navigation command
        explanation = cmd.get('explanation', '').lower()
        raw_lower   = text.lower()
        for zone, (zx, zy, _) in HOSPITAL_ZONES.items():
            if zone in raw_lower or zone in explanation:
                self.get_logger().info(f'Navigating to zone: {zone} ({zx}, {zy})')
                threading.Thread(
                    target=self._navigate_to_zone,
                    args=(zone, zx, zy),
                    daemon=True
                ).start()
                return

        self.get_logger().info(f'Command: {cmd.get("explanation","")}')

        lx  = float(cmd.get('linear_x',  0.0))
        az  = float(cmd.get('angular_z', 0.0))
        dur = cmd.get('duration')

        if dur:
            threading.Thread(
                target=self._timed, args=(lx, az, float(dur)), daemon=True
            ).start()
        else:
            self._pub(lx, az)

    def _navigate_to_zone(self, zone_name, tx, ty):
        """Simple navigate to zone using /cmd_vel — move forward then adjust."""
        import math
        self.get_logger().info(f'Going to {zone_name} at ({tx}, {ty})')
        # This is a simple open-loop navigation
        # For full navigation use Nav2 — this demo uses timed movements
        self._pub(0.2, 0.0)
        time.sleep(3.0)
        self._pub(0.0, 0.0)
        self.get_logger().info(f'Arrived at {zone_name}')

    def _timed(self, lx, az, dur):
        self.get_logger().info(f'Timed {dur:.1f}s: linear={lx} angular={az}')
        self._pub(lx, az)
        time.sleep(dur)
        self._pub(0.0, 0.0)
        self.get_logger().info('Timed done')

    def _pub(self, lx, az):
        msg = Twist()
        msg.linear.x  = float(lx)
        msg.angular.z = float(az)
        self.cmd_pub.publish(msg)
        self.get_logger().info(f'→ linear={lx:.2f} angular={az:.2f}')


def main():
    rclpy.init()
    node = VoiceControl()
    rclpy.spin(node)
    node.stream.stop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
