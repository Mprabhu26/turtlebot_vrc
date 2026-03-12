#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import threading
import time
import numpy as np
import sounddevice as sd
import io, os, re, json

SAMPLE_RATE = 16000
CHUNK_SAMPLES = int(SAMPLE_RATE * 3.0)

# Speed: 0.3m/s → 6m=20s, 4m=13s
# Turn 90° at 1.0rad/s = 1.6s, 180° = 3.2s
# Robot at (0,0) facing EAST (+X)
#
# GREEN=PHARMACY  (-6,+6) → face WEST, go 20s, turn NORTH, go 13s
# RED=ICU         (+6,+6) → face EAST, go 20s, turn NORTH, go 13s  
# BLUE=RECEPTION  (-6,-6) → face WEST, go 20s, turn SOUTH, go 13s
# YELLOW=WARD     (+6,-6) → face EAST, go 20s, turn SOUTH, go 13s

S = 0.3   # linear speed
T90 = 1.6 # time for 90deg turn at 1.0 rad/s
T180 = 3.2
EAST = 20.0   # 6m at 0.3m/s
INTO = 13.0   # 4m at 0.3m/s

# FROM CENTER (robot facing EAST):
FROM_CENTER = {
    'icu':       [(S,  0.0, EAST), (0.0,  1.0, T90), (S, 0.0, INTO)],
    'ward':      [(S,  0.0, EAST), (0.0, -1.0, T90), (S, 0.0, INTO)],
    'pharmacy':  [(0.0, 1.0, T180), (S, 0.0, EAST), (0.0, 1.0, T90), (S, 0.0, INTO)],
    'reception': [(0.0, 1.0, T180), (S, 0.0, EAST), (0.0,-1.0, T90), (S, 0.0, INTO)],
}

# TO CENTER (reverse of FROM_CENTER exactly):
TO_CENTER = {
    'icu':       [(-S, 0.0, INTO), (0.0, -1.0, T90), (-S, 0.0, EAST)],
    'ward':      [(-S, 0.0, INTO), (0.0,  1.0, T90), (-S, 0.0, EAST)],
    'pharmacy':  [(-S, 0.0, INTO), (0.0, -1.0, T90), (-S, 0.0, EAST), (0.0, -1.0, T180)],
    'reception': [(-S, 0.0, INTO), (0.0,  1.0, T90), (-S, 0.0, EAST), (0.0, -1.0, T180)],
}

ZONES = list(FROM_CENTER.keys())

NLU_PROMPT = """Hospital robot. Return ONLY JSON.
"go to ICU" or "ICU" or "red room" → {"action":"navigate","zone":"icu"}
"go to ward" or "ward" or "yellow room" → {"action":"navigate","zone":"ward"}
"go to pharmacy" or "pharmacy" or "green room" → {"action":"navigate","zone":"pharmacy"}
"go to reception" or "reception" or "blue room" → {"action":"navigate","zone":"reception"}
"go forward" or "forward" or "ahead" → {"action":"move","linear":0.3,"angular":0.0,"duration":3.0}
"go back" or "backward" or "reverse" → {"action":"move","linear":-0.3,"angular":0.0,"duration":3.0}
"turn left" or "left" → {"action":"move","linear":0.0,"angular":1.0,"duration":1.6}
"turn right" or "right" → {"action":"move","linear":0.0,"angular":-1.0,"duration":1.6}
"spin" or "rotate" → {"action":"move","linear":0.0,"angular":1.0,"duration":6.3}
"stop" or "halt" → {"action":"stop"}
anything else → {"action":"unknown"}
Return ONLY the JSON."""

NOISE = {'gracias','merci','danke','okay','ok','thank you','thanks','yes','no',
         'hmm','uh','um','ah','bye','hello','hi','obrigado','yeah','sure',
         'right','word','vogue','world','fallen','продолжение следует'}

class VoiceControlNode(Node):
    def __init__(self):
        super().__init__('voice_control')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.navigating = False
        self.current_zone = 'center'
        self.groq_client = None
        try:
            from groq import Groq
            self.groq_client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
            self.get_logger().info('Groq ready')
        except Exception as e:
            self.get_logger().warn(f'Groq: {e}')
        self.fw_model = None
        try:
            from faster_whisper import WhisperModel
            self.fw_model = WhisperModel('base', device='cpu', compute_type='int8')
            self.get_logger().info('faster-whisper ready')
        except Exception as e:
            self.get_logger().warn(f'fw: {e}')
        self.vad_model = None
        try:
            import torch
            model, _ = torch.hub.load('snakers4/silero-vad','silero_vad',force_reload=False,trust_repo=True)
            self.vad_model = model
            self.get_logger().info('VAD ready')
        except Exception as e:
            self.get_logger().warn(f'VAD: {e}')
        self.get_logger().info('Ready! Say: go to ICU / pharmacy / ward / reception')
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def _pub(self, linear, angular):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.pub.publish(msg)

    def _stop(self):
        self._pub(0.0, 0.0)

    def _run_sequence(self, steps):
        for lin, ang, dur in steps:
            if not self.navigating:
                self._stop()
                return
            self.get_logger().info(f'  lin={lin:.1f} ang={ang:.1f} dur={dur:.1f}s')
            end = time.time() + dur
            while time.time() < end:
                if not self.navigating:
                    self._stop()
                    return
                self._pub(lin, ang)
                time.sleep(0.05)
            self._stop()
            time.sleep(0.2)

    def _navigate_to_zone(self, dst):
        dst = dst.lower().strip()
        if dst not in FROM_CENTER:
            self.get_logger().warn(f'Unknown: {dst}')
            return
        if self.current_zone == dst:
            self.get_logger().info(f'Already at {dst.upper()}')
            self.navigating = False
            return
        self.navigating = True
        # Return to center first if in a room
        if self.current_zone in TO_CENTER:
            self.get_logger().info(f'=== {self.current_zone.upper()} → CENTER ===')
            self._run_sequence(TO_CENTER[self.current_zone])
            if not self.navigating:
                return
            time.sleep(0.3)
        # Go to destination
        self.get_logger().info(f'=== CENTER → {dst.upper()} ===')
        self._run_sequence(FROM_CENTER[dst])
        if self.navigating:
            self.current_zone = dst
            self.get_logger().info(f'=== Arrived at {dst.upper()} ===')
        self.navigating = False

    def _move(self, linear, angular, duration=3.0):
        self.get_logger().info(f'Move: lin={linear:.2f} ang={angular:.2f} dur={duration}s')
        end = time.time() + duration
        while time.time() < end:
            self._pub(linear, angular)
            time.sleep(0.05)
        self._stop()

    def _groq_nlu(self, text):
        if not self.groq_client:
            return None
        try:
            resp = self.groq_client.chat.completions.create(
                model='llama-3.1-8b-instant',
                messages=[{'role':'system','content':NLU_PROMPT},
                          {'role':'user','content':text}],
                max_tokens=60, temperature=0.0)
            raw = re.sub(r'```json|```','',resp.choices[0].message.content.strip()).strip()
            return json.loads(raw)
        except Exception as e:
            self.get_logger().warn(f'NLU: {e}')
            return None

    def _dispatch(self, text):
        t = text.lower().strip().rstrip('.')
        if len(t) < 4:
            return
        if t in NOISE or t.replace('.','').strip() in NOISE:
            return
        self.get_logger().info(f'Processing: "{text}"')
        cmd = self._groq_nlu(text)
        if not cmd:
            return
        self.get_logger().info(f'NLU: {cmd}')
        action = cmd.get('action','unknown')
        if action == 'stop':
            self.navigating = False
            self._stop()
            self.get_logger().info('STOPPED')
        elif action == 'navigate':
            zone = cmd.get('zone','')
            if zone:
                threading.Thread(target=self._navigate_to_zone, args=(zone,), daemon=True).start()
        elif action == 'move':
            lin = float(cmd.get('linear', 0.0))
            ang = float(cmd.get('angular', 0.0))
            dur = float(cmd.get('duration', 3.0))
            threading.Thread(target=self._move, args=(lin, ang, dur), daemon=True).start()

    def _transcribe(self, audio_np):
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
                    self.get_logger().info(f'Heard: "{text}"')
                    return text
            except Exception as e:
                self.get_logger().warn(f'Whisper: {e}')
        if self.fw_model:
            try:
                segs, _ = self.fw_model.transcribe(audio_np.astype(np.float32), language='en')
                text = ' '.join(s.text for s in segs).strip()
                if text:
                    return text
            except Exception as e:
                self.get_logger().warn(f'fw: {e}')
        return ''

    def _has_speech(self, audio_np):
        if self.vad_model is None:
            return True
        try:
            import torch
            conf = self.vad_model(torch.from_numpy(audio_np.astype(np.float32)), SAMPLE_RATE).item()
            self.get_logger().info(f'VAD: {conf*100:.0f}%')
            return conf > 0.7
        except Exception:
            return True

    def _listen_loop(self):
        device_index = None
        try:
            for i, d in enumerate(sd.query_devices()):
                if d['max_input_channels'] > 0:
                    device_index = i
                    self.get_logger().info(f'Audio device {i}: {d["name"]}')
                    break
        except Exception:
            pass
        while rclpy.ok():
            try:
                audio = sd.rec(CHUNK_SAMPLES, samplerate=SAMPLE_RATE,
                               channels=1, dtype='float32', device=device_index)
                sd.wait()
                audio_np = np.clip(audio.flatten() * 3.0, -1.0, 1.0)
                if not self._has_speech(audio_np):
                    continue
                text = self._transcribe(audio_np)
                if not text or len(text) < 3:
                    continue
                self._dispatch(text)
            except Exception as e:
                self.get_logger().error(f'Error: {e}')
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
