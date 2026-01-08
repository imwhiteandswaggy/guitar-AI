"""
Guitar Teacher - Geometric String Detection
Automatically calculates string positions from neck detection
No calibration needed - plug and play!
"""

import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import time
from collections import deque
from string_refinement import refine_string_positions_with_edges

# ============================================================================
# CONFIGURATION
# ============================================================================

FRET_MODEL = "trained_models/real_guitar_test3/weights/best.pt"
CONFIDENCE = 0.5

# Standard tuning (from high to low)
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
OPEN_STRINGS = {
    1: 4,   # High E (string 1)
    2: 11,  # B (string 2)
    3: 7,   # G (string 3)
    4: 2,   # D (string 4)
    5: 9,   # A (string 5)
    6: 4    # Low E (string 6)
}

# Colors
COLOR_STRING = (0, 255, 255)      # Cyan
COLOR_FRET = (0, 255, 0)          # Green
COLOR_NECK = (255, 165, 0)        # Orange
COLOR_FINGER = (255, 0, 255)      # Magenta
COLOR_NOTE = (255, 255, 0)        # Yellow
COLOR_TEXT = (255, 255, 255)      # White

# ============================================================================
# GUITAR TEACHER CLASS
# ============================================================================

class GeometricGuitarTeacher:
    def __init__(self):
        print("🎸 Initializing Geometric Guitar Teacher...")
        
        # Load models
        self.fret_model = YOLO(FRET_MODEL)
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # State
        self.running = True
        self.show_strings = True
        self.show_frets = True
        self.fps_history = deque(maxlen=30)
        self.frame_count = 0
        
        print("✓ Models loaded!")
        print("✓ Using GEOMETRIC string detection (no calibration needed!)\n")
    
    def calculate_string_positions(self, neck_box, nut_box=None):
        """
        Calculate 6 string Y-positions using tapered spacing model.
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
        taper_factor = 0.2
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
        """Determine which string based on Y coordinate"""
        if not string_positions:
            return None
        
        # Find closest string
        min_dist = float('inf')
        closest_string = None
        neck_height = string_positions[-1] - string_positions[0]
        threshold = neck_height / 12  # Half the spacing between strings
        
        for i, string_y in enumerate(string_positions):
            dist = abs(y_pos - string_y)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                closest_string = i + 1
        
        return closest_string
    
    def get_note_name(self, string_num, fret_num):
        """Calculate note name from string and fret"""
        if string_num not in OPEN_STRINGS:
            return None
        
        open_note_index = OPEN_STRINGS[string_num]
        note_index = (open_note_index + fret_num) % 12
        return NOTES[note_index]
    
    def map_frets_to_numbers(self, fret_boxes, neck_box):
        """Map detected fret boxes to fret numbers (RIGHT to LEFT)"""
        if not fret_boxes or neck_box is None:
            return []
        
        sorted_frets = sorted(fret_boxes, key=lambda box: box[0])
        total_frets = len(sorted_frets)
        fret_map = []
        
        for idx, box in enumerate(sorted_frets):
            fret_number = total_frets - idx
            fret_map.append((fret_number, box))
        
        return fret_map
    
    def get_fret_from_position(self, x_pos, fret_map, neck_box):
        """Determine which fret an x-position is on"""
        if not fret_map or neck_box is None:
            return None
        
        neck_left = neck_box[0]
        sorted_map = sorted(fret_map, key=lambda x: x[0])
        
        # Check if in nut area (open string)
        if sorted_map:
            first_fret_right = sorted_map[0][1][2]
            if neck_left < x_pos < first_fret_right:
                return 0
        
        # Check each fret
        for fret_num, (x1, y1, x2, y2) in sorted_map:
            if x1 <= x_pos <= x2:
                return fret_num
        
        # Between frets
        for i in range(len(sorted_map) - 1):
            curr_fret, curr_box = sorted_map[i]
            next_fret, next_box = sorted_map[i + 1]
            if curr_box[2] < x_pos < next_box[0]:
                return next_fret
        
        return None
    
    def draw_ui_overlay(self, frame):
        """Draw UI overlay"""
        h, w = frame.shape[:2]
        
        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 80), COLOR_STRING, 2)
        
        cv2.putText(frame, "GUITAR TEACHER - GEOMETRIC", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_STRING, 2)
        cv2.putText(frame, "Auto string detection (no calibration!)", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
        
        # FPS
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (w - 120, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
        
        # Help text
        cv2.putText(frame, "S: Toggle strings | F: Toggle frets | Q: Quit",
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
    
    def process_frame(self, frame):
        """Process a single frame"""
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect frets, neck, and nut
        fret_results = self.fret_model(frame, conf=CONFIDENCE, verbose=False)
        fret_detections = fret_results[0].boxes
        
        fret_boxes = []
        neck_box = None
        nut_box = None
        
        for box in fret_detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = self.fret_model.names[int(box.cls[0])]
            
            if class_name == "fret":
                fret_boxes.append((x1, y1, x2, y2))
                if self.show_frets:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_FRET, 2)
            elif class_name == "neck":
                neck_box = (x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_NECK, 3)
                cv2.putText(frame, "NECK", (x1 + 10, y1 + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_NECK, 2)
            elif class_name == "nut":
                nut_box = (x1, y1, x2, y2)
                if self.show_frets:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
        
        # Calculate string positions with tapered spacing
        string_positions = self.calculate_string_positions(neck_box, nut_box)
        
        # Refine positions using edge detection
        if string_positions and neck_box:
            string_positions = refine_string_positions_with_edges(
                frame, neck_box, string_positions
            )
        
        # Draw strings
        if self.show_strings and string_positions:
            string_names = ['e (high)', 'B', 'G', 'D', 'A', 'E (low)']
            for i, string_y in enumerate(string_positions):
                cv2.line(frame, (0, string_y), (w, string_y), COLOR_STRING, 2)
                cv2.putText(frame, f"S{i+1}", (w - 50, string_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_STRING, 1)
        
        # Map frets
        fret_map = self.map_frets_to_numbers(fret_boxes, neck_box)
        
        # Draw fret numbers
        if self.show_frets:
            for fret_num, (fx1, fy1, fx2, fy2) in fret_map:
                label_pos = ((fx1 + fx2) // 2 - 10, fy1 - 10)
                cv2.putText(frame, str(fret_num), label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_FRET, 2)
        
        # Detect hands
        hand_results = self.hands.process(frame_rgb)
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Draw hand skeleton
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Check fingertips
                fingertip_ids = [8, 12, 16, 20]
                finger_names = ["Index", "Middle", "Ring", "Pinky"]
                
                for tip_id, finger_name in zip(fingertip_ids, finger_names):
                    landmark = hand_landmarks.landmark[tip_id]
                    fx = int(landmark.x * w)
                    fy = int(landmark.y * h)
                    
                    # Get string and fret
                    string_num = self.get_string_from_y(fy, string_positions)
                    fret_num = self.get_fret_from_position(fx, fret_map, neck_box)
                    
                    if string_num and fret_num is not None:
                        # Calculate note
                        note_name = self.get_note_name(string_num, fret_num)
                        
                        # Draw fingertip
                        cv2.circle(frame, (fx, fy), 12, COLOR_FINGER, -1)
                        cv2.circle(frame, (fx, fy), 14, COLOR_TEXT, 2)
                        
                        # Label with note
                        if fret_num == 0:
                            label = f"{finger_name}: S{string_num} Open = {note_name}"
                        else:
                            label = f"{finger_name}: S{string_num} F{fret_num} = {note_name}"
                        
                        # Draw label with background
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (fx + 15, fy - th - 5), 
                                    (fx + 15 + tw, fy + 5), (0, 0, 0), -1)
                        cv2.putText(frame, label, (fx + 20, fy),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_NOTE, 2)
        
        # Draw UI
        self.draw_ui_overlay(frame)
        
        return frame
    
    def run(self):
        """Main teaching loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Could not open webcam")
            return
        
        print("="*60)
        print("🎸 GUITAR TEACHER - READY!")
        print("="*60)
        print("Controls:")
        print("  S - Toggle string overlay")
        print("  F - Toggle fret boxes")
        print("  Q - Quit")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        while True:
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = self.process_frame(frame)
            self.frame_count += 1
            
            # Calculate FPS
            frame_time = time.time() - frame_start
            fps = 1.0 / frame_time if frame_time > 0 else 0
            self.fps_history.append(fps)
            
            cv2.imshow('Guitar Teacher', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.show_strings = not self.show_strings
            elif key == ord('f'):
                self.show_frets = not self.show_frets
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("✓ Session complete!")
        print(f"Total frames: {self.frame_count}")
        print(f"Duration: {int(time.time() - start_time)}s")
        if self.fps_history:
            print(f"Average FPS: {sum(self.fps_history)/len(self.fps_history):.1f}")
        print("="*60)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    teacher = GeometricGuitarTeacher()
    teacher.run()
