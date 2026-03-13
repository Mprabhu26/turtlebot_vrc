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

# CONFIRMED LAYOUT (top-down view):
# RED    = top-left     = reception
# YELLOW = top-right    = icu
# GREEN  = bottom-left  = ward
# BLUE   = bottom-right = pharmacy
#
# Robot faces SOUTH at spawn
# S=0.3m/s, T90=1.6s turn 90deg, T180=3.2s turn 180deg
# CORR=17s along corridor, INTO=10s into room

S=0.3; T90=1.6; T180=3.2; CORR=17.0; INTO=10.0

# Building blocks:
# go_north  = turn 180 (face north) then go CORR
# go_south  = go CORR (already facing south)
# turn_left_enter  = turn left then INTO
# turn_right_enter = turn right then INTO
# exit_north_left  = reverse INTO, turn right, reverse CORR, turn 180 (back to center facing south)
# exit_north_right = reverse INTO, turn left,  reverse CORR, turn 180
# exit_south_left  = reverse INTO, turn right, reverse CORR
# exit_south_right = reverse INTO, turn left,  reverse CORR

# DIRECT PATHS — hardcoded for every combination
PATHS = {
    # ── FROM CENTER ──────────────────────────────────────────────
    ('center','reception'): [
        (0.0, 1.0,T180),(S,0.0,CORR),(0.0,1.0,T90),(S,0.0,INTO)],   # N then left→RED
    ('center','icu'): [
        (0.0,1.0,T180),(S,0.0,CORR),(0.0,-1.0,T90),(S,0.0,INTO)],   # N then right→YELLOW
    ('center','ward'): [
        (S,0.0,CORR),(0.0,-1.0,T90),(S,0.0,INTO)],                   # S then right→GREEN
    ('center','pharmacy'): [
        (S,0.0,CORR),(0.0,1.0,T90),(S,0.0,INTO)],                    # S then left→BLUE

    # ── FROM RECEPTION (RED, top-left, entered facing WEST) ──────
    # Exit: back east INTO, turn south (right), back CORR north, turn south (face south=center)
    ('reception','center'): [
        (-S,0.0,INTO),(0.0,-1.0,T90),(-S,0.0,CORR),(0.0,1.0,T180)],
    ('reception','icu'): [                                             # RED→YELLOW: exit to center, go to icu
        (-S,0.0,INTO),(0.0,-1.0,T90),(-S,0.0,CORR),                  # back to corridor facing south
        (0.0,1.0,T180),(S,0.0,CORR),(0.0,-1.0,T90),(S,0.0,INTO)],   # go north, turn right
    ('reception','ward'): [                                            # RED→GREEN
        (-S,0.0,INTO),(0.0,-1.0,T90),(-S,0.0,CORR),                  # back to corridor facing south
        (S,0.0,CORR),(0.0,-1.0,T90),(S,0.0,INTO)],                   # go south, turn right
    ('reception','pharmacy'): [                                        # RED→BLUE
        (-S,0.0,INTO),(0.0,-1.0,T90),(-S,0.0,CORR),                  # back to corridor facing south
        (S,0.0,CORR),(0.0,1.0,T90),(S,0.0,INTO)],                    # go south, turn left

    # ── FROM ICU (YELLOW, top-right, entered facing EAST) ────────
    # Exit: back west INTO, turn south (left), back CORR, now facing south
    ('icu','center'): [
        (-S,0.0,INTO),(0.0,1.0,T90),(-S,0.0,CORR),(0.0,1.0,T180)],
    ('icu','reception'): [                                             # YELLOW→RED
        (-S,0.0,INTO),(0.0,1.0,T90),(-S,0.0,CORR),                   # back to corridor facing south
        (0.0,1.0,T180),(S,0.0,CORR),(0.0,1.0,T90),(S,0.0,INTO)],    # go north, turn left
    ('icu','ward'): [                                                  # YELLOW→GREEN
        (-S,0.0,INTO),(0.0,1.0,T90),(-S,0.0,CORR),                   # back to corridor facing south
        (S,0.0,CORR),(0.0,-1.0,T90),(S,0.0,INTO)],                   # go south, turn right
    ('icu','pharmacy'): [                                              # YELLOW→BLUE
        (-S,0.0,INTO),(0.0,1.0,T90),(-S,0.0,CORR),                   # back to corridor facing south
        (S,0.0,CORR),(0.0,1.0,T90),(S,0.0,INTO)],                    # go south, turn left

    # ── FROM WARD (GREEN, bottom-left, entered facing WEST) ──────
    # Exit: back east INTO, turn north (left), back CORR south, now facing south
    ('ward','center'): [
        (-S,0.0,INTO),(0.0,1.0,T90),(-S,0.0,CORR)],
    ('ward','reception'): [                                            # GREEN→RED
        (-S,0.0,INTO),(0.0,1.0,T90),(-S,0.0,CORR),                   # back to corridor facing south
        (0.0,1.0,T180),(S,0.0,CORR),(0.0,1.0,T90),(S,0.0,INTO)],    # go north, turn left
    ('ward','icu'): [                                                  # GREEN→YELLOW
        (-S,0.0,INTO),(0.0,1.0,T90),(-S,0.0,CORR),                   # back to corridor facing south
        (0.0,1.0,T180),(S,0.0,CORR),(0.0,-1.0,T90),(S,0.0,INTO)],   # go north, turn right
    ('ward','pharmacy'): [                                             # GREEN→BLUE
        (-S,0.0,INTO),(0.0,1.0,T90),(-S,0.0,CORR),                   # back to corridor facing south
        (S,0.0,CORR),(0.0,1.0,T90),(S,0.0,INTO)],                    # go south, turn left

    # ── FROM PHARMACY (BLUE, bottom-right, entered facing EAST) ──
    # Exit: back west INTO, turn north (right), back CORR south, now facing south
    ('pharmacy','center'): [
        (-S,0.0,INTO),(0.0,-1.0,T90),(-S,0.0,CORR)],
    ('pharmacy','reception'): [                                        # BLUE→RED
        (-S,0.0,INTO),(0.0,-1.0,T90),(-S,0.0,CORR),                  # back to corridor facing south
        (0.0,1.0,T180),(S,0.0,CORR),(0.0,1.0,T90),(S,0.0,INTO)],    # go north, turn left
    ('pharmacy','icu'): [                                              # BLUE→YELLOW
        (-S,0.0,INTO),(0.0,-1.0,T90),(-S,0.0,CORR),                  # back to corridor facing south
        (0.0,1.0,T180),(S,0.0,CORR),(0.0,-1.0,T90),(S,0.0,INTO)],   # go north, turn right
    ('pharmacy','ward'): [                                             # BLUE→GREEN
        (-S,0.0,INTO),(0.0,-1.0,T90),(-S,0.0,CORR),                  # back to corridor facing south
        (S,0.0,CORR),(0.0,-1.0,T90),(S,0.0,INTO)],                   # go south, turn right
}

NLU_PROMPT = """Hospital robot. Return ONLY JSON.
"go to ICU" or "ICU" or "yellow" → {"action":"navigate","zone":"icu"}
"go to pharmacy" or "pharmacy" or "blue" → {"action":"navigate","zone":"pharmacy"}
"go to reception" or "reception" or "red" → {"action":"navigate","zone":"reception"}
"go to ward" or "ward" or "green" → {"action":"navigate","zone":"ward"}
"forward" or "go forward" → {"action":"move","linear":0.3,"angular":0.0,"duration":3.0}
"backward" or "go back" → {"action":"move","linear":-0.3,"angular":0.0,"duration":3.0}
"turn left" or "left" → {"action":"move","linear":0.0,"angular":1.0,"duration":1.6}
"turn right" or "right" → {"action":"move","linear":0.0,"angular":-1.0,"duration":1.6}
"spin" or "rotate" → {"action":"move","linear":0.0,"angular":1.0,"duration":6.3}
"stop" or "halt" → {"action":"stop"}
anything else → {"action":"unknown"}
Return ONLY the JSON."""

NOISE = {'gracias','merci','danke','okay','ok','thank you','thanks','yes','no',
         'hmm','uh','um','ah','bye','hello','hi','obrigado','yeah','sure',
         'word','vogue','world','fallen','go to'}

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
                self._stop(); return
            self.get_logger().info(f'  lin={lin:.1f} ang={ang:.1f} dur={dur:.1f}s')
            end = time.time() + dur
            while time.time() < end:
                if not self.navigating:
                    self._stop(); return
                self._pub(lin, ang)
                time.sleep(0.05)
            self._stop()
            time.sleep(0.15)

    def _navigate_to_zone(self, dst):
        dst = dst.lower().strip()
        zones = ['reception','icu','ward','pharmacy']
        if dst not in zones:
            self.get_logger().warn(f'Unknown: {dst}'); return
        if self.current_zone == dst:
            self.get_logger().info(f'Already at {dst.upper()}')
            self.navigating = False; return
        key = (self.current_zone, dst)
        if key not in PATHS:
            self.get_logger().warn(f'No path: {key}'); return
        self.navigating = True
        self.get_logger().info(f'=== {self.current_zone.upper()} → {dst.upper()} ===')
        self._run_sequence(PATHS[key])
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
            self.get_logger().warn(f'NLU: {e}'); return None

    def _dispatch(self, text):
        t = text.lower().strip().rstrip('.')
        if len(t) < 4: return
        if t in NOISE or t.replace('.','').strip() in NOISE: return
        self.get_logger().info(f'Processing: "{text}"')
        cmd = self._groq_nlu(text)
        if not cmd: return
        self.get_logger().info(f'NLU: {cmd}')
        action = cmd.get('action','unknown')
        if action == 'stop':
            self.navigating = False; self._stop()
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
                if text: return text
            except: pass
        return ''

    def _has_speech(self, audio_np):
        if self.vad_model is None: return True
        try:
            import torch
            conf = self.vad_model(torch.from_numpy(audio_np.astype(np.float32)), SAMPLE_RATE).item()
            self.get_logger().info(f'VAD: {conf*100:.0f}%')
            return conf > 0.7
        except: return True

    def _listen_loop(self):
        device_index = None
        try:
            for i, d in enumerate(sd.query_devices()):
                if d['max_input_channels'] > 0:
                    device_index = i
                    self.get_logger().info(f'Audio device {i}: {d["name"]}')
                    break
        except: pass
        while rclpy.ok():
            try:
                audio = sd.rec(CHUNK_SAMPLES, samplerate=SAMPLE_RATE,
                               channels=1, dtype='float32', device=device_index)
                sd.wait()
                audio_np = np.clip(audio.flatten() * 3.0, -1.0, 1.0)
                if not self._has_speech(audio_np): continue
                text = self._transcribe(audio_np)
                if not text or len(text) < 3: continue
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
