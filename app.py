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
import base64
import threading
import queue
import sounddevice as sd
import librosa
from chord_library import CHORD_LIBRARY, BEGINNER_CHORDS, get_chord_info, evaluate_chord
from string_refinement import refine_string_positions_with_edges

# ============================================================================
# CONFIGURATION
# ============================================================================

FRET_MODEL = "trained_models/real_guitar_test3/weights/best.pt"
CONFIDENCE = 0.5
SAMPLE_RATE = 22050
BUFFER_SIZE = 2048
HOP_LENGTH = 512

# Standard tuning
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
                    y=audio_mono,
                    sr=SAMPLE_RATE,
                    hop_length=HOP_LENGTH,
                    fmin=80,
                    fmax=1000
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
    
    def start(self):
        self.running = True
        self.stream = sd.InputStream(
            device=2,  # Webcam mic
            callback=self.audio_callback,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=BUFFER_SIZE
        )
        self.stream.start()
        self.thread = threading.Thread(target=self.process_audio, daemon=True)
        self.thread.start()
    
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
        self.fret_model = YOLO(FRET_MODEL)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.audio = AudioDetector()
        self.audio.start()
        
        print("✓ Detection engine ready!")
    
    def calculate_string_positions(self, neck_box, nut_box=None):
        """
        Calculate string positions using tapered spacing model.
        Strings are wider at the nut and narrower at the bridge.
        """
        if neck_box is None:
            return []
        
        x1, y1, x2, y2 = neck_box
        neck_bottom = y2
        
        # Use nut top as anchor point if available, otherwise use neck top
        if nut_box:
            nut_y_top = nut_box[1]
        else:
            nut_y_top = y1
        
        neck_height = neck_bottom - nut_y_top
        
        if neck_height <= 0:
            return []
        
        # Tapered spacing: wider at nut, narrower at bridge
        # Typical guitar taper factor: 0.15-0.25
        taper_factor = 0.2
        
        # Spacing calculations
        spacing_at_nut = neck_height / 5.5  # Wider spacing at nut
        spacing_at_bridge = neck_height / 6.5  # Narrower spacing at bridge
        
        string_positions = []
        cumulative_y = 0
        
        for i in range(6):
            # Progress from nut (0.0) to bridge (1.0)
            progress = (i + 0.5) / 6.0
            
            # Interpolate spacing based on taper
            current_spacing = spacing_at_nut * (1 - taper_factor * progress)
            
            # First string: half spacing from nut
            if i == 0:
                string_y = nut_y_top + current_spacing / 2
                cumulative_y = current_spacing / 2
            else:
                # Add spacing for this string
                cumulative_y += current_spacing
                string_y = nut_y_top + cumulative_y
            
            string_positions.append(int(string_y))
        
        return string_positions
    
    def get_string_from_y(self, y_pos, string_positions):
        if not string_positions:
            return None
        
        min_dist = float('inf')
        closest_string = None
        neck_height = string_positions[-1] - string_positions[0]
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
    
    def process_frame(self, frame):
        """Process frame and return detection data"""
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect frets, neck, and nut
        fret_results = self.fret_model(frame, conf=CONFIDENCE, verbose=False)
        fret_boxes = []
        neck_box = None
        nut_box = None
        
        for box in fret_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = self.fret_model.names[int(box.cls[0])]
            
            if class_name == "fret":
                fret_boxes.append((x1, y1, x2, y2))
            elif class_name == "neck":
                neck_box = (x1, y1, x2, y2)
            elif class_name == "nut":
                nut_box = (x1, y1, x2, y2)
        
        # Calculate string positions with tapered spacing
        string_positions = self.calculate_string_positions(neck_box, nut_box)
        
        # Refine positions using edge detection
        if string_positions and neck_box:
            string_positions = refine_string_positions_with_edges(
                frame, neck_box, string_positions
            )
        fret_map = self.map_frets_to_numbers(fret_boxes, neck_box)
        
        # Detect hands and notes
        hand_results = self.hands.process(frame_rgb)
        detected_notes = []
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                fingertip_ids = [4, 8, 12, 16, 20]
                finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                
                for tip_id, finger_name in zip(fingertip_ids, finger_names):
                    landmark = hand_landmarks.landmark[tip_id]
                    fx = int(landmark.x * w)
                    fy = int(landmark.y * h)
                    
                    if not self.is_on_neck(fx, fy, neck_box):
                        continue
                    
                    string_num = self.get_string_from_y(fy, string_positions)
                    fret_num = self.get_fret_from_position(fx, fret_map, neck_box)
                    
                    if string_num and fret_num is not None:
                        vision_note = self.get_note_name(string_num, fret_num)
                        audio_note = self.audio.detected_note
                        
                        detected_notes.append({
                            'finger': finger_name,
                            'string': string_num,
                            'fret': fret_num,
                            'note': vision_note,
                            'x': fx,
                            'y': fy,
                            'matches_audio': vision_note == audio_note if audio_note else None
                        })
        
        return {
            'notes': detected_notes,
            'audio_note': self.audio.detected_note,
            'audio_freq': float(self.audio.detected_freq) if self.audio.detected_freq else 0,
            'neck_detected': neck_box is not None,
            'frets_detected': len(fret_boxes)
        }

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)
CORS(app)

# Global detection engine
engine = None
camera = None
current_mode = "free_play"  # Modes: "free_play", "chord_trainer"
current_chord = "E Minor"  # Default chord for training

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera

def generate_frames():
    """Generate video frames with detection overlay"""
    global engine
    
    if engine is None:
        engine = GuitarDetectionEngine()
    
    cam = get_camera()
    
    while True:
        success, frame = cam.read()
        if not success:
            break
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Serve the web UI"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_data')
def detection_data():
    """Get current detection data as JSON"""
    global engine
    
    if engine is None:
        engine = GuitarDetectionEngine()
    
    cam = get_camera()
    success, frame = cam.read()
    
    if not success:
        return jsonify({'error': 'No frame available'})
    
    data = engine.process_frame(frame)
    
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
    """Switch between modes"""
    global current_mode
    
    if mode in ['free_play', 'chord_trainer']:
        current_mode = mode
        return jsonify({'success': True, 'mode': current_mode})
    else:
        return jsonify({'success': False, 'error': 'Invalid mode'})

@app.route('/set_chord/<chord_name>')
def set_chord(chord_name):
    """Set the current chord for training"""
    global current_chord
    
    if chord_name in CHORD_LIBRARY:
        current_chord = chord_name
        return jsonify({'success': True, 'chord': current_chord})
    else:
        return jsonify({'success': False, 'error': 'Unknown chord'})

@app.route('/get_chords')
def get_chords():
    """Get list of available chords"""
    return jsonify({
        'chords': list(CHORD_LIBRARY.keys()),
        'beginner_chords': BEGINNER_CHORDS
    })

if __name__ == '__main__':
    print("="*60)
    print("🎸 Starting Guitar Teacher Web App")
    print("="*60)
    print("Open your browser to: http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
