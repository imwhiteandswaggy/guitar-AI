"""
Guitar Teacher - Web Backend
Flask server that provides webcam feed and detection data to the web UI
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import threading
import queue
import os
import socket
import time
import sounddevice as sd
import librosa
from chord_library import CHORD_LIBRARY, BEGINNER_CHORDS, get_chord_info, evaluate_chord
from string_refinement import refine_string_positions_with_edges
from chord_overlay import render_chord_overlay
from string_tracking import HybridTracker

# ============================================================================
# CONFIGURATION
# ============================================================================

FRET_MODEL = "trained_models/real_guitar_test3/weights/best.pt"
CONFIDENCE = 0.25
SAMPLE_RATE = 22050
BUFFER_SIZE = 2048
HOP_LENGTH = 512

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
OPEN_STRINGS = {
    1: 4,   # High E
    2: 11,  # B
    3: 7,   # G
    4: 2,   # D
    5: 9,   # A
    6: 4    # Low E
}

# ============================================================================
# AUDIO DETECTION
# ============================================================================

class AudioDetector:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.detected_note = None
        self.detected_freq = 0
        self.confidence = 0
        self.running = False
        
    def freq_to_note(self, frequency):
        if frequency < 50 or frequency > 2000:
            return None
        note_number = 12 * np.log2(frequency / 440.0) + 69
        note_number = int(round(note_number))
        note_index = note_number % 12
        return NOTES[note_index]
    
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())
    
    def process_audio(self):
        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                audio_mono = np.mean(audio_data, axis=1) if len(audio_data.shape) > 1 else audio_data
                
                pitches, magnitudes = librosa.piptrack(
                    y=audio_mono, sr=SAMPLE_RATE,
                    hop_length=HOP_LENGTH, fmin=80, fmax=1000
                )
                
                index = magnitudes.argmax()
                pitch = pitches[index // pitches.shape[1], index % pitches.shape[1]]
                magnitude = magnitudes[index // magnitudes.shape[1], index % magnitudes.shape[1]]
                
                if magnitude > 0.1 and pitch > 0:
                    self.detected_freq = pitch
                    self.detected_note = self.freq_to_note(pitch)
                    self.confidence = min(magnitude * 10, 1.0)
                else:
                    self.detected_note = None
                    self.confidence = 0
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio processing error: {e}")
    
    def start(self, device=None):
        """Start audio detection with auto-detection of device."""
        self.running = True
        
        if device is None:
            try:
                default_device = sd.default.device[0]
                if default_device is not None:
                    device = default_device
                    print(f"  Audio device: {device}")
                else:
                    devices = sd.query_devices()
                    for i, dev in enumerate(devices):
                        if dev['max_input_channels'] > 0:
                            device = i
                            print(f"  Audio device: {i} ({dev['name']})")
                            break
            except Exception as e:
                print(f"  Audio disabled: {e}")
                self.running = False
                return
        
        try:
            self.stream = sd.InputStream(
                device=device, callback=self.audio_callback,
                channels=1, samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE
            )
            self.stream.start()
            self.thread = threading.Thread(target=self.process_audio, daemon=True)
            self.thread.start()
        except Exception as e:
            print(f"  Audio disabled: {e}")
            self.running = False
    
    def stop(self):
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

# ============================================================================
# DETECTION ENGINE
# ============================================================================

class GuitarDetectionEngine:
    def __init__(self):
        print("🎸 Initializing detection engine...")
        
        if not os.path.exists(FRET_MODEL):
            raise FileNotFoundError(f"Model not found: {FRET_MODEL}. Run: python download_model.py")
        
        self.fret_model = YOLO(FRET_MODEL)
        print(f"  Model classes: {list(self.fret_model.names.values())}")
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        
        self.audio = AudioDetector()
        self.audio.start(device=None)
        
        self.calibrated_strings = None
        self.string_tracker = HybridTracker(alpha=0.25, window_size=5, max_change=25)
        
        # Shared detection state (written by generate_frames, read by detection_data)
        self.last_detection = {
            'notes': [],
            'audio_note': None,
            'audio_freq': 0,
            'neck_detected': False,
            'frets_detected': 0
        }
        self._debug_counter = 0
        
        print("✓ Detection engine ready!")
    
    def calculate_string_positions(self, neck_box, nut_box=None, playable_y_min=None, playable_y_max=None):
        """
        Calculate string positions using tapered spacing model.
        Distributes 6 string Ys within [playable_y_min, playable_y_max] when given,
        otherwise within nut_y_top to bottom of neck.
        """
        if neck_box is None:
            return []
        
        x1, y1, x2, y2 = neck_box
        nut_y_top = nut_box[1] if nut_box else y1
        neck_height = y2 - nut_y_top
        
        if neck_height <= 0:
            return []
        
        # Explicit bounds so the band never exceeds neck/fretboard
        top = playable_y_min if playable_y_min is not None else nut_y_top
        bottom = playable_y_max if playable_y_max is not None else min(nut_y_top + neck_height, y2)
        span = bottom - top
        if span <= 0:
            return []
        
        taper_factor = 0.2
        spacing_at_nut = span / 5.5
        
        string_positions = []
        cumulative_y = 0
        
        for i in range(6):
            progress = (i + 0.5) / 6.0
            current_spacing = spacing_at_nut * (1 - taper_factor * progress)
            
            if i == 0:
                string_y = top + current_spacing / 2
                cumulative_y = current_spacing / 2
            else:
                cumulative_y += current_spacing
                string_y = top + cumulative_y
            
            string_positions.append(int(string_y))
        
        return string_positions
    
    def get_string_from_y(self, y_pos, string_positions):
        if not string_positions or not isinstance(string_positions, list):
            return None
        
        min_dist = float('inf')
        closest_string = None
        neck_height = string_positions[-1] - string_positions[0]
        if neck_height <= 0:
            return None
        threshold = neck_height / 12
        
        for i, string_y in enumerate(string_positions):
            dist = abs(y_pos - string_y)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                closest_string = i + 1
        
        return closest_string
    
    def get_note_name(self, string_num, fret_num):
        if string_num not in OPEN_STRINGS:
            return None
        open_note_index = OPEN_STRINGS[string_num]
        note_index = (open_note_index + fret_num) % 12
        return NOTES[note_index]
    
    def map_frets_to_numbers(self, fret_boxes, neck_box):
        if not fret_boxes or neck_box is None:
            return []
        sorted_frets = sorted(fret_boxes, key=lambda box: box[0])
        return [(len(sorted_frets) - idx, box) for idx, box in enumerate(sorted_frets)]
    
    def get_fret_from_position(self, x_pos, fret_map, neck_box):
        if not fret_map or neck_box is None:
            return None
        
        neck_left = neck_box[0]
        sorted_map = sorted(fret_map, key=lambda x: x[0])
        
        if sorted_map:
            first_fret_right = sorted_map[0][1][2]
            if neck_left < x_pos < first_fret_right:
                return 0
        
        for fret_num, (x1, y1, x2, y2) in sorted_map:
            if x1 <= x_pos <= x2:
                return fret_num
        
        for i in range(len(sorted_map) - 1):
            curr_fret, curr_box = sorted_map[i]
            next_fret, next_box = sorted_map[i + 1]
            if curr_box[2] < x_pos < next_box[0]:
                return next_fret
        
        return None
    
    def is_on_neck(self, x, y, neck_box):
        if neck_box is None:
            return False
        nx1, ny1, nx2, ny2 = neck_box
        margin = 30
        return (nx1 - margin) <= x <= (nx2 + margin) and (ny1 - margin) <= y <= (ny2 + margin)

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

# Global state
engine = None
camera = None
camera_lock = threading.Lock()  # Thread-safe camera access
current_camera_id = 0
current_mode = "free_play"
current_chord = "E Minor"
overlay_enabled = True
debug_mode = False
show_raw_feed = False

# Neck detection temporal smoothing
last_neck_box = None
neck_box_history = []
MAX_NECK_HISTORY = 5

# ============================================================================
# CAMERA MANAGEMENT
# ============================================================================

def get_available_cameras():
    """Detect available camera devices."""
    import sys
    backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY
    cameras = []
    for i in range(5):
        cap = None
        try:
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                # Quick warm-up so read() returns real frame
                for _ in range(5):
                    cap.read()
                ret, _ = cap.read()
                if ret:
                    cameras.append({'id': i, 'name': f'Camera {i}', 'available': True})
        except Exception:
            continue
        finally:
            if cap is not None:
                cap.release()
    
    if not cameras:
        cameras.append({'id': 0, 'name': 'Default Camera', 'available': True})
    
    return cameras

def get_camera(camera_id=None):
    """Get camera instance with error handling and fallbacks."""
    global camera, current_camera_id
    
    with camera_lock:
        if camera_id is not None:
            current_camera_id = camera_id
            if camera is not None:
                camera.release()
            camera = None
        
        if camera is None:
            try:
                # On Windows, DirectShow often gives a visible feed; default can be black
                import sys
                if sys.platform == 'win32':
                    camera = cv2.VideoCapture(current_camera_id, cv2.CAP_DSHOW)
                else:
                    camera = cv2.VideoCapture(current_camera_id)
                
                if not camera.isOpened():
                    camera = cv2.VideoCapture(current_camera_id)  # try default backend
                if not camera.isOpened():
                    if current_camera_id != 0:
                        camera.release()
                        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform == 'win32' else 0)
                        current_camera_id = 0
                    
                    if not camera.isOpened():
                        print("⚠ No camera available")
                        return None
                
                # Set camera properties before reading (some drivers need this first)
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
                camera.set(cv2.CAP_PROP_FPS, 30)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Warm-up: discard first 20 frames (many cameras output black initially)
                for _ in range(20):
                    camera.read()
                
                # Test frame read
                ret, test_frame = camera.read()
                if not ret or test_frame is None:
                    camera.release()
                    camera = None
                    return None
                
                w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"✓ Camera {current_camera_id}: {w}x{h}")
                
            except Exception as e:
                print(f"⚠ Camera error: {e}")
                camera = None
        
        return camera

# ============================================================================
# PLAYABLE REGION (string geometry bounds)
# ============================================================================

def get_playable_y_bounds(neck_box, nut_box, fret_boxes):
    """
    Vertical range where strings should lie (fretboard).
    Option B: union of fret boxes and nut; fallback to neck box.
    Returns (y_min, y_max) or (None, None) if neck_box is None.
    """
    if neck_box is None:
        return None, None
    y1_neck, y2_neck = neck_box[1], neck_box[3]
    y_min, y_max = y1_neck, y2_neck
    if nut_box is not None:
        y_min = min(y_min, nut_box[1])
        y_max = max(y_max, nut_box[3])
    for box in (fret_boxes or []):
        if len(box) >= 4:
            y_min = min(y_min, box[1])
            y_max = max(y_max, box[3])
    margin = 3
    return y_min - margin, y_max + margin


def clamp_string_positions_to_bounds(string_positions, y_min, y_max):
    """Clamp each of the 6 string Y values to [y_min, y_max]. Returns list of ints."""
    if not string_positions or len(string_positions) != 6 or y_min is None or y_max is None:
        return string_positions
    return [max(y_min, min(y_max, int(y))) for y in string_positions]


# ============================================================================
# VIDEO STREAM GENERATOR
# ============================================================================

def generate_frames():
    """Generate video frames with detection overlay."""
    global engine, current_chord, overlay_enabled, debug_mode, show_raw_feed
    global last_neck_box, neck_box_history
    
    # Lazy-init engine
    if engine is None:
        try:
            engine = GuitarDetectionEngine()
        except Exception as e:
            print(f"⚠ Engine init failed: {e}")
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, f"Init failed: {str(e)[:50]}", (10, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return
    
    cam = get_camera()
    if cam is None:
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "No camera available", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return
    
    frame_skip = 0
    max_frame_skip = 1
    detection_counter = 0
    reconnect_failures = 0
    
    # Local state for string positions (persists across frames)
    string_positions = []
    fret_map = []
    
    while True:
        try:
            # Thread-safe camera read
            with camera_lock:
                if cam is None or not cam.isOpened():
                    cam = get_camera()
                    if cam is None:
                        break
                success, frame = cam.read()
            
            if not success or frame is None:
                reconnect_failures += 1
                if reconnect_failures <= 3:
                    print(f"⚠ Camera read failed (attempt {reconnect_failures})")
                time.sleep(0.5)
                cam = get_camera()
                if cam is None and reconnect_failures > 10:
                    break
                continue
            
            reconnect_failures = 0
            
            # Resize frame early for performance
            h, w = frame.shape[:2]
            if max(h, w) > 960:
                scale = 960 / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
                h, w = frame.shape[:2]
            
            # Raw feed mode - skip all processing
            if show_raw_feed:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                continue
            
            # Frame skipping - send fast frames without detection
            frame_skip += 1
            if frame_skip <= max_frame_skip:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                continue
            
            frame_skip = 0
            detection_counter += 1
            
            # === DETECTION PHASE ===
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Adaptive confidence based on prior detection
            detection_conf = CONFIDENCE
            if last_neck_box is not None:
                detection_conf = max(0.15, CONFIDENCE * 0.6)
            
            fret_results = engine.fret_model(frame, conf=detection_conf, verbose=False)
            
            fret_boxes = []
            neck_box = None
            nut_box = None
            
            # Parse YOLO results
            if fret_results and len(fret_results) > 0:
                for box in fret_results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = engine.fret_model.names[int(box.cls[0])]
                    
                    if class_name == "fret":
                        fret_boxes.append((x1, y1, x2, y2))
                    elif class_name == "neck":
                        neck_box = (x1, y1, x2, y2)
                        neck_box_history.append(neck_box)
                        if len(neck_box_history) > MAX_NECK_HISTORY:
                            neck_box_history.pop(0)
                        last_neck_box = neck_box
                    elif class_name == "nut":
                        nut_box = (x1, y1, x2, y2)
            
            # Temporal smoothing: use cached neck if current detection failed
            if neck_box is None and last_neck_box is not None and len(neck_box_history) > 0:
                avg_neck = tuple(
                    int(sum(box[i] for box in neck_box_history) / len(neck_box_history))
                    for i in range(4)
                )
                neck_box = avg_neck
            
            # Debug logging (every ~2 seconds)
            engine._debug_counter += 1
            if engine._debug_counter % 60 == 0:
                n = len(fret_results[0].boxes) if fret_results and len(fret_results) > 0 else 0
                if n > 0:
                    classes = [engine.fret_model.names[int(b.cls[0])] for b in fret_results[0].boxes]
                    print(f"🔍 {n} detections: {classes}")
                else:
                    print("⚠ No detections")
            
            # === STRING POSITIONS ===
            fret_map = engine.map_frets_to_numbers(fret_boxes, neck_box) if neck_box else []
            playable_y_min, playable_y_max = get_playable_y_bounds(neck_box, nut_box, fret_boxes)

            if engine.calibrated_strings:
                string_positions = engine.calibrated_strings
            else:
                string_positions = engine.calculate_string_positions(
                    neck_box, nut_box,
                    playable_y_min=playable_y_min, playable_y_max=playable_y_max
                )
            
            # Ensure correct order
            if string_positions and len(string_positions) == 6:
                if string_positions[-1] < string_positions[0]:
                    string_positions = string_positions[::-1]
            
            if string_positions and neck_box:
                if not engine.calibrated_strings:
                    refined_positions, confidence_scores = refine_string_positions_with_edges(
                        frame, neck_box, string_positions, fret_map=fret_map
                    )
                    
                    if len(refined_positions) == 6 and refined_positions[-1] < refined_positions[0]:
                        refined_positions = refined_positions[::-1]
                        confidence_scores = confidence_scores[::-1]
                    
                    string_positions = engine.string_tracker.update(
                        refined_positions, confidence_scores=confidence_scores,
                        y_min=playable_y_min, y_max=playable_y_max
                    )
            else:
                prev = engine.string_tracker.get_positions()
                if prev:
                    string_positions = prev
            
            # Clamp string positions to playable region so they never draw outside the neck
            if string_positions and playable_y_min is not None and playable_y_max is not None:
                string_positions = clamp_string_positions_to_bounds(
                    string_positions, playable_y_min, playable_y_max
                )
            
            # === UPDATE SHARED DETECTION STATE ===
            # (read by /detection_data without running the pipeline again)
            audio_note = engine.audio.detected_note
            detected_notes = []
            
            if neck_box and string_positions and detection_counter % 2 == 0:
                hand_results_for_data = engine.hands.process(frame_rgb)
                if hand_results_for_data and hand_results_for_data.multi_hand_landmarks:
                    for hand_landmarks in hand_results_for_data.multi_hand_landmarks:
                        for tip_id, finger_name in zip([4, 8, 12, 16, 20], ["Thumb", "Index", "Middle", "Ring", "Pinky"]):
                            lm = hand_landmarks.landmark[tip_id]
                            fx, fy = int(lm.x * w), int(lm.y * h)
                            if not engine.is_on_neck(fx, fy, neck_box):
                                continue
                            s = engine.get_string_from_y(fy, string_positions)
                            f = engine.get_fret_from_position(fx, fret_map, neck_box)
                            if s and f is not None:
                                note = engine.get_note_name(s, f)
                                detected_notes.append({
                                    'finger': finger_name, 'string': s, 'fret': f,
                                    'note': note, 'x': fx, 'y': fy,
                                    'matches_audio': note == audio_note if audio_note else None
                                })
            
            engine.last_detection = {
                'notes': detected_notes,
                'audio_note': audio_note,
                'audio_freq': float(engine.audio.detected_freq) if engine.audio.detected_freq else 0,
                'neck_detected': neck_box is not None,
                'frets_detected': len(fret_boxes)
            }
            
            # === DRAWING PHASE ===
            
            # Status text
            if neck_box is None:
                cv2.putText(frame, "No neck detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"Neck detected | Frets: {len(fret_boxes)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Debug mode drawing
            if debug_mode:
                if neck_box:
                    bx1, by1, bx2, by2 = neck_box
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
                    cv2.putText(frame, "NECK", (bx1 + 5, by1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if nut_box:
                    bx1, by1, bx2, by2 = nut_box
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 165, 255), 3)
                    cv2.putText(frame, "NUT", (bx1 + 5, by1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                for i, (bx1, by1, bx2, by2) in enumerate(fret_boxes):
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
                    cv2.putText(frame, f"F{i+1}", (bx1 + 5, by1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                if string_positions and len(string_positions) == 6 and neck_box:
                    bx1, by1, bx2, by2 = neck_box
                    for i, sy in enumerate(string_positions):
                        cv2.line(frame, (bx1, sy), (bx2, sy), (255, 255, 0), 2)
                        cv2.putText(frame, f"S{i+1}", (bx1 + 5, sy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Hand skeleton
                hand_results_debug = engine.hands.process(frame_rgb)
                hand_count = 0
                if hand_results_debug and hand_results_debug.multi_hand_landmarks:
                    hand_count = len(hand_results_debug.multi_hand_landmarks)
                    for hand_landmarks in hand_results_debug.multi_hand_landmarks:
                        engine.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, engine.mp_hands.HAND_CONNECTIONS,
                            engine.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                            engine.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                        )
                        for tip_id, fname in zip([4, 8, 12, 16, 20], ["Thumb", "Index", "Middle", "Ring", "Pinky"]):
                            lm = hand_landmarks.landmark[tip_id]
                            fx, fy = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame, (fx, fy), 8, (0, 255, 255), -1)
                            cv2.putText(frame, fname, (fx + 10, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                            if neck_box and engine.is_on_neck(fx, fy, neck_box):
                                cv2.putText(frame, "ON NECK", (fx + 10, fy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                
                info_y = 60
                cv2.putText(frame, "DEBUG MODE", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Neck: {'YES' if neck_box else 'NO'}", (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Frets: {len(fret_boxes)}", (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Nut: {'YES' if nut_box else 'NO'}", (10, info_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Hands: {hand_count}", (10, info_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Chord overlay
            if not debug_mode and overlay_enabled and current_chord and neck_box and string_positions:
                chord_info = get_chord_info(current_chord)
                if chord_info:
                    detected_fingers = [n for n in detected_notes if 'finger' in n]
                    
                    if neck_box and string_positions and len(string_positions) == 6:
                        bx1, by1, bx2, by2 = neck_box
                        names = ['E1', 'B2', 'G3', 'D4', 'A5', 'E6']
                        for i, sy in enumerate(string_positions):
                            cv2.line(frame, (bx1, sy), (bx2, sy), (255, 255, 0), 2)
                            cv2.putText(frame, f"S{i+1} ({names[i]})", (bx1 + 5, sy - 8),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
                    frame = render_chord_overlay(
                        frame, chord_info, neck_box, fret_map, string_positions,
                        detected_fingers=detected_fingers, show_strings=False
                    )
            
            # Encode and yield frame
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        except Exception as e:
            print(f"⚠ Frame error: {e}")
            continue

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={'Cache-Control': 'no-cache, no-store, must-revalidate', 'Pragma': 'no-cache', 'Expires': '0'}
    )

@app.route('/detection_data')
def detection_data():
    """Return cached detection data (computed by generate_frames, not re-run)."""
    global engine
    
    if engine is None:
        return jsonify({'notes': [], 'audio_note': None, 'audio_freq': 0, 'neck_detected': False, 'frets_detected': 0})
    
    data = dict(engine.last_detection)
    
    # Add mode-specific data
    if current_mode == "chord_trainer":
        chord_info = get_chord_info(current_chord)
        evaluation = evaluate_chord(data['notes'], current_chord) if data['notes'] else None
        data['mode'] = 'chord_trainer'
        data['target_chord'] = chord_info
        data['evaluation'] = evaluation
    else:
        data['mode'] = 'free_play'
    
    return jsonify(data)

@app.route('/set_mode/<mode>')
def set_mode(mode):
    global current_mode
    if mode in ['free_play', 'chord_trainer']:
        current_mode = mode
        return jsonify({'success': True, 'mode': current_mode})
    return jsonify({'success': False, 'error': 'Invalid mode'})

@app.route('/set_chord/<chord_name>')
def set_chord(chord_name):
    global current_chord
    if chord_name in CHORD_LIBRARY:
        current_chord = chord_name
        return jsonify({'success': True, 'chord': current_chord})
    return jsonify({'success': False, 'error': 'Unknown chord'})

@app.route('/get_chords')
def get_chords():
    return jsonify({'chords': list(CHORD_LIBRARY.keys()), 'beginner_chords': BEGINNER_CHORDS})

@app.route('/api/cameras')
def api_cameras():
    cameras = get_available_cameras()
    return jsonify({'cameras': cameras, 'current': current_camera_id})

@app.route('/api/set_camera/<int:camera_id>', methods=['POST'])
def api_set_camera(camera_id):
    global camera, current_camera_id
    try:
        test_cap = cv2.VideoCapture(camera_id)
        if test_cap.isOpened():
            ret, _ = test_cap.read()
            test_cap.release()
            if ret:
                current_camera_id = camera_id
                with camera_lock:
                    if camera is not None:
                        camera.release()
                    camera = None
                get_camera()
                return jsonify({'success': True, 'camera_id': camera_id})
        return jsonify({'success': False, 'error': 'Camera not available'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/toggle_overlay')
def api_toggle_overlay():
    global overlay_enabled
    overlay_enabled = not overlay_enabled
    return jsonify({'success': True, 'enabled': overlay_enabled})

@app.route('/api/overlay_status')
def api_overlay_status():
    return jsonify({'enabled': overlay_enabled, 'chord': current_chord})

@app.route('/api/calibrate_strings', methods=['POST'])
def api_calibrate_strings():
    global engine
    if engine is None:
        return jsonify({'success': False, 'error': 'Engine not initialized'})
    try:
        data = request.get_json()
        positions = data.get('positions', [])
        if len(positions) != 6:
            return jsonify({'success': False, 'error': 'Must provide 6 string positions'})
        engine.calibrated_strings = sorted([int(pos) for pos in positions])
        return jsonify({'success': True, 'positions': engine.calibrated_strings})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/reset_calibration', methods=['POST'])
def api_reset_calibration():
    global engine
    if engine:
        engine.calibrated_strings = None
    return jsonify({'success': True, 'message': 'Calibration reset'})

@app.route('/api/toggle_debug', methods=['POST'])
def api_toggle_debug():
    global debug_mode
    debug_mode = not debug_mode
    return jsonify({'success': True, 'debug_mode': debug_mode})

@app.route('/api/toggle_raw_feed', methods=['POST'])
def api_toggle_raw_feed():
    global show_raw_feed
    show_raw_feed = not show_raw_feed
    return jsonify({'success': True, 'raw_feed': show_raw_feed})

@app.route('/api/debug_status')
def api_debug_status():
    return jsonify({'debug_mode': debug_mode, 'raw_feed': show_raw_feed})

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    port = 5000
    
    # Find a free port
    for try_port in [5000, 5001, 8080, 8081]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', try_port))
        sock.close()
        if result != 0:  # Port is free
            port = try_port
            break
    
    print("=" * 60)
    print("🎸 Guitar Teacher AI")
    print("=" * 60)
    print(f"✓ Open browser to: http://127.0.0.1:{port}")
    print(f"✓ Or: http://localhost:{port}")
    print("=" * 60)
    
    try:
        app.run(debug=False, host='127.0.0.1', port=port, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n✓ Server stopped")
    except Exception as e:
        print(f"\n✗ Server error: {e}")
