"""
Guitar Teacher - Minimal Clean UI
Focus on note detection, minimal visual clutter
"""

import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import time
from collections import deque
from string_refinement import refine_string_positions_with_edges
from chord_overlay import render_chord_overlay, draw_chord_info_overlay
from chord_library import CHORD_LIBRARY, BEGINNER_CHORDS, get_chord_info
from string_calibration import manual_string_calibration

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
COLOR_FINGER = (255, 0, 255)      # Magenta
COLOR_NOTE = (0, 255, 0)          # Green
COLOR_TEXT = (255, 255, 255)      # White
COLOR_BG = (0, 0, 0)              # Black

# ============================================================================
# GUITAR TEACHER CLASS
# ============================================================================

class MinimalGuitarTeacher:
    def __init__(self):
        print("🎸 Initializing Minimal Guitar Teacher...")
        
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
        self.show_hand_skeleton = True
        self.show_debug = False
        self.fps_history = deque(maxlen=30)
        self.frame_count = 0
        
        # Chord overlay state
        self.overlay_enabled = True
        self.current_chord_index = 0
        self.current_chord = BEGINNER_CHORDS[0] if BEGINNER_CHORDS else None
        self.show_strings = True  # Show by default for debugging
        
        # String calibration
        self.calibrated_strings = None  # Manually calibrated positions
        
        print("✓ Models loaded!")
        print("✓ Clean minimal UI - focus on notes!\n")
    
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
    
    def is_position_on_neck(self, x, y, neck_box):
        """Check if position is within neck boundaries"""
        if neck_box is None:
            return False
        
        nx1, ny1, nx2, ny2 = neck_box
        margin = 30
        return (nx1 - margin) <= x <= (nx2 + margin) and (ny1 - margin) <= y <= (ny2 + margin)
    
    def draw_ui_overlay(self, frame, notes_detected):
        """Draw minimal UI overlay"""
        h, w = frame.shape[:2]
        
        # Top bar - very minimal
        cv2.rectangle(frame, (0, 0), (w, 80), COLOR_BG, -1)
        cv2.rectangle(frame, (0, 0), (w, 80), COLOR_NOTE, 1)
        
        cv2.putText(frame, "GUITAR TEACHER", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_NOTE, 2)
        
        # Chord info overlay
        if self.current_chord:
            draw_chord_info_overlay(frame, self.current_chord, self.overlay_enabled, x=20, y=55)
        
        # FPS
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            cv2.putText(frame, f"{avg_fps:.0f} FPS", (w - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
        
        # Note count
        if notes_detected > 0:
            cv2.circle(frame, (w - 30, 25), 15, COLOR_NOTE, -1)
            cv2.putText(frame, str(notes_detected), (w - 38, 33),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BG, 2)
        
        # Bottom help
        help_text = "D: Debug | H: Hand | O: Overlay | C: Chord | S: Strings | K: Calibrate | Q: Quit"
        cv2.putText(frame, help_text, (20, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Show calibration status
        if self.calibrated_strings:
            cv2.putText(frame, "CALIBRATED", (w - 150, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
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
                # Optional: very faint fret markers in debug mode
                if self.show_debug:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 0), 1)
            elif class_name == "neck":
                neck_box = (x1, y1, x2, y2)
                # Optional: very faint neck outline in debug mode
                if self.show_debug:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
            elif class_name == "nut":
                nut_box = (x1, y1, x2, y2)
                # Optional: very faint nut outline in debug mode
                if self.show_debug:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 1)
        
        # Calculate string positions
        if self.calibrated_strings:
            # Use manually calibrated positions
            string_positions = self.calibrated_strings
        else:
            # Calculate with tapered spacing
            string_positions = self.calculate_string_positions(neck_box, nut_box)
            
            # Ensure string positions are ordered correctly (string 1 at top = smaller Y)
            if string_positions and len(string_positions) == 6:
                # Always sort so smallest Y is first (string 1 = high E = top)
                # This handles cases where neck_box might be inverted
                sorted_with_indices = sorted(enumerate(string_positions), key=lambda x: x[1])
                # Reorder to maintain string order (0-5) but ensure Y increases
                if sorted_with_indices[0][0] != 0:
                    # Positions are not in order, need to reorder
                    # Find which index corresponds to which string
                    reordered = [0] * 6
                    for orig_idx, y_pos in sorted_with_indices:
                        # Map original index to new position
                        reordered[orig_idx] = y_pos
                    # If first string (index 0) is not at top, reverse
                    if sorted_with_indices[0][0] == 5:
                        # Last string is at top, reverse everything
                        string_positions = string_positions[::-1]
                        print("Warning: String positions were inverted, corrected")
                # Final check: ensure first Y < last Y
                if string_positions[-1] < string_positions[0]:
                    string_positions = string_positions[::-1]
                    print("Warning: String positions were inverted (final check), corrected")
            
            # Refine positions using improved multi-method approach
            if string_positions and neck_box:
                fret_map = self.map_frets_to_numbers(fret_boxes, neck_box)
                string_positions = refine_string_positions_with_edges(
                    frame, neck_box, string_positions, fret_map=fret_map
                )
                
                # Ensure still ordered correctly after refinement
                if len(string_positions) == 6 and string_positions[-1] < string_positions[0]:
                    string_positions = string_positions[::-1]
        
        # Map frets
        fret_map = self.map_frets_to_numbers(fret_boxes, neck_box)
        
        # Debug: Draw string positions with labels
        if string_positions and neck_box and (self.show_debug or self.show_strings):
            x1, y1, x2, y2 = neck_box
            string_names = ['E1', 'B2', 'G3', 'D4', 'A5', 'E6']
            for i, string_y in enumerate(string_positions):
                # Draw line
                cv2.line(frame, (x1, string_y), (x2, string_y), (0, 255, 255), 1)
                # Label
                cv2.putText(frame, f"S{i+1}", (x1 + 5, string_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Detect hands
        hand_results = self.hands.process(frame_rgb)
        notes_detected = 0
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Optional: Draw hand skeleton
                if self.show_hand_skeleton:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                # Check fingertips
                fingertip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
                finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                
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
                        notes_detected += 1
                        
                        # Draw fingertip - clean circle
                        cv2.circle(frame, (fx, fy), 15, COLOR_FINGER, -1)
                        cv2.circle(frame, (fx, fy), 17, COLOR_TEXT, 2)
                        
                        # Note label - clean and readable
                        if fret_num == 0:
                            label = f"{note_name}"
                            sublabel = f"S{string_num} Open"
                        else:
                            label = f"{note_name}"
                            sublabel = f"S{string_num} F{fret_num}"
                        
                        # Main note label (big and clear)
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                        
                        # Background box
                        padding = 10
                        box_x1 = fx + 25
                        box_y1 = fy - th - padding
                        box_x2 = fx + 25 + tw + padding * 2
                        box_y2 = fy + padding
                        
                        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), COLOR_BG, -1)
                        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), COLOR_NOTE, 2)
                        
                        # Note name
                        cv2.putText(frame, label, (fx + 35, fy),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_NOTE, 3)
                        
                        # Sublabel (smaller)
                        cv2.putText(frame, sublabel, (fx + 35, fy + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Render chord overlay if enabled
        if self.overlay_enabled and self.current_chord:
            chord_info = get_chord_info(self.current_chord)
            if chord_info:
                # Collect detected fingers for feedback
                detected_fingers = []
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        fingertip_ids = [4, 8, 12, 16, 20]
                        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                        
                        for tip_id, finger_name in zip(fingertip_ids, finger_names):
                            landmark = hand_landmarks.landmark[tip_id]
                            fx = int(landmark.x * w)
                            fy = int(landmark.y * h)
                            
                            if not self.is_position_on_neck(fx, fy, neck_box):
                                continue
                            
                            string_num = self.get_string_from_y(fy, string_positions)
                            fret_num = self.get_fret_from_position(fx, fret_map, neck_box)
                            
                            if string_num and fret_num is not None:
                                detected_fingers.append({
                                    'finger': finger_name,
                                    'string': string_num,
                                    'fret': fret_num
                                })
                
                # Render overlay
                frame = render_chord_overlay(
                    frame, chord_info, neck_box, fret_map, string_positions,
                    detected_fingers=detected_fingers, show_strings=self.show_strings
                )
        
        # Draw UI
        self.draw_ui_overlay(frame, notes_detected)
        
        return frame
    
    def run(self):
        """Main teaching loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Could not open webcam")
            return
        
        print("="*60)
        print("🎸 GUITAR TEACHER - MINIMAL UI")
        print("="*60)
        print("Controls:")
        print("  D - Toggle debug overlay (show frets/neck)")
        print("  H - Toggle hand skeleton")
        print("  O - Toggle chord overlay")
        print("  C - Next chord in progression")
        print("  S - Toggle string lines")
        print("  K - Manual string calibration (click on strings)")
        print("  R - Reset calibration (use auto-detection)")
        print("  Q - Quit")
        print("="*60)
        if self.current_chord:
            print(f"Current chord: {self.current_chord}")
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
            elif key == ord('d'):
                self.show_debug = not self.show_debug
                print(f"Debug mode: {'ON' if self.show_debug else 'OFF'}")
            elif key == ord('h'):
                self.show_hand_skeleton = not self.show_hand_skeleton
                print(f"Hand skeleton: {'ON' if self.show_hand_skeleton else 'OFF'}")
            elif key == ord('o'):
                self.overlay_enabled = not self.overlay_enabled
                print(f"Chord overlay: {'ON' if self.overlay_enabled else 'OFF'}")
            elif key == ord('c'):
                if BEGINNER_CHORDS:
                    self.current_chord_index = (self.current_chord_index + 1) % len(BEGINNER_CHORDS)
                    self.current_chord = BEGINNER_CHORDS[self.current_chord_index]
                    print(f"Switched to chord: {self.current_chord}")
            elif key == ord('s'):
                self.show_strings = not self.show_strings
                print(f"String lines: {'ON' if self.show_strings else 'OFF'}")
            elif key == ord('k'):
                # Manual calibration
                if neck_box is None:
                    print("Error: Neck not detected. Please ensure guitar is visible.")
                else:
                    print("Starting manual string calibration...")
                    print("Click on each string (1-6, top to bottom) in the calibration window")
                    print("Press ESC to cancel, ENTER when done")
                    ret, cal_frame = cap.read()
                    if ret:
                        calibrated = manual_string_calibration(cal_frame.copy(), neck_box)
                        if calibrated:
                            self.calibrated_strings = calibrated
                            print(f"Calibration complete! String positions: {calibrated}")
                        else:
                            print("Calibration cancelled")
            elif key == ord('r'):
                # Reset calibration
                self.calibrated_strings = None
                print("Calibration reset - using auto-detection")
        
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
    teacher = MinimalGuitarTeacher()
    teacher.run()
