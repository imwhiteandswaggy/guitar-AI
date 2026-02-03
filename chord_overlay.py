"""
Chord Overlay Renderer
Renders AR-style chord diagrams directly onto the guitar neck
Shows dots, finger numbers, and note names at target positions
"""

import cv2
import numpy as np


# Finger name to number mapping
FINGER_MAP = {
    "Index": 1,
    "Middle": 2,
    "Ring": 3,
    "Pinky": 4,
    "Thumb": 0  # Usually not used for fretting
}

# Colors for each finger (BGR format for OpenCV)
FINGER_COLORS = {
    1: (0, 100, 255),      # Blue (Index)
    2: (0, 255, 0),        # Green (Middle)
    3: (0, 255, 255),       # Yellow (Ring)
    4: (0, 0, 255),        # Red (Pinky)
    0: (128, 128, 128)     # Gray (Thumb/unused)
}

# Standard tuning note mapping
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
OPEN_STRINGS = {
    1: 4,   # High E
    2: 11,  # B
    3: 7,   # G
    4: 2,   # D
    5: 9,   # A
    6: 4    # Low E
}


def get_note_name(string_num, fret_num):
    """Calculate note name from string and fret number"""
    if string_num not in OPEN_STRINGS:
        return None
    
    open_note_index = OPEN_STRINGS[string_num]
    note_index = (open_note_index + fret_num) % 12
    return NOTES[note_index]


def apply_perspective_correction(x, y, neck_box, fret_map):
    """
    Apply perspective correction for camera angle
    
    Args:
        x, y: Original coordinates
        neck_box: Neck bounding box
        fret_map: Fret map for reference
    
    Returns:
        Corrected (x, y) coordinates
    """
    if neck_box is None:
        return x, y
    
    x1, y1, x2, y2 = neck_box
    neck_width = x2 - x1
    neck_height = y2 - y1
    
    # Estimate perspective angle from fret alignment
    # If frets are tilted, apply correction
    if fret_map and len(fret_map) >= 2:
        sorted_frets = sorted(fret_map, key=lambda f: f[0])
        if len(sorted_frets) >= 2:
            # Get first and last fret
            first_fret = sorted_frets[0][1]
            last_fret = sorted_frets[-1][1]
            
            # Calculate tilt angle
            first_y_top = first_fret[1]
            first_y_bottom = first_fret[3]
            last_y_top = last_fret[1]
            last_y_bottom = last_fret[3]
            
            # Average tilt
            tilt_top = (last_y_top - first_y_top) / neck_width if neck_width > 0 else 0
            tilt_bottom = (last_y_bottom - first_y_bottom) / neck_width if neck_width > 0 else 0
            avg_tilt = (tilt_top + tilt_bottom) / 2
            
            # Apply correction (small adjustments)
            if abs(avg_tilt) > 0.01:  # Only if significant tilt
                # Normalize x position (0 to 1)
                x_norm = (x - x1) / neck_width if neck_width > 0 else 0.5
                # Apply tilt correction
                y_correction = avg_tilt * (x - x1) * 0.3  # Dampen effect
                y = int(y + y_correction)
    
    return x, y


def calculate_position_pixel(string_num, fret_num, neck_box, fret_map, string_positions):
    """
    Convert (string, fret) to pixel coordinates on frame
    
    Args:
        string_num: String number (1-6, 1=high E)
        fret_num: Fret number (0=open, 1+ = fretted)
        neck_box: Tuple (x1, y1, x2, y2) of neck bounding box
        fret_map: List of (fret_number, (x1, y1, x2, y2)) tuples
        string_positions: List of 6 Y-coordinates for strings
    
    Returns:
        Tuple (x, y) pixel coordinates, or None if invalid
    """
    if not string_positions or len(string_positions) < string_num:
        return None
    
    if neck_box is None:
        return None
    
    # Get string Y position
    string_y = string_positions[string_num - 1]
    
    # Get fret X position using improved multi-method approach
    if fret_num == 0:  # Open string
        # Position at nut area (left side of neck)
        neck_left = neck_box[0]
        if fret_map:
            # Position between nut and first fret
            sorted_map = sorted(fret_map, key=lambda x: x[0])
            if sorted_map:
                first_fret_box = sorted_map[0][1]
                first_fret_right = first_fret_box[2] if isinstance(first_fret_box, tuple) else first_fret_box[2]
                fret_x = neck_left + (first_fret_right - neck_left) * 0.3
            else:
                fret_x = neck_left + 20
        else:
            fret_x = neck_left + 20
    else:
        # Improved fret position calculation using multiple methods
        fret_x = None
        methods = []
        
        if fret_map:
            sorted_map = sorted(fret_map, key=lambda x: x[0])
            
            # Method 1: Use center of fret box
            for mapped_fret_num, fret_box in sorted_map:
                if mapped_fret_num == fret_num:
                    center_x = (fret_box[0] + fret_box[2]) // 2
                    methods.append(center_x)
                    
                    # Method 2: Use left edge + width/2 (more accurate for perspective)
                    fret_width = fret_box[2] - fret_box[0]
                    left_center_x = fret_box[0] + fret_width // 2
                    methods.append(left_center_x)
                    break
            
            # Method 3: Interpolate from neighboring frets
            if not methods:
                # Find closest frets
                for i, (mapped_fret_num, fret_box) in enumerate(sorted_map):
                    if mapped_fret_num > fret_num:
                        # Found next fret
                        if i > 0:
                            prev_fret_num, prev_fret_box = sorted_map[i-1]
                            # Interpolate between previous and current
                            prev_center = (prev_fret_box[0] + prev_fret_box[2]) // 2
                            curr_center = (fret_box[0] + fret_box[2]) // 2
                            
                            # Linear interpolation
                            ratio = (fret_num - prev_fret_num) / (mapped_fret_num - prev_fret_num)
                            interpolated_x = prev_center + (curr_center - prev_center) * ratio
                            methods.append(int(interpolated_x))
                        break
        
        # Use average of all methods for robustness
        if methods:
            fret_x = int(np.mean(methods))
        else:
            # Fallback: Estimate position
            neck_left = neck_box[0]
            neck_right = neck_box[2]
            neck_width = neck_right - neck_left
            
            if fret_map:
                sorted_map = sorted(fret_map, key=lambda x: x[0])
                if sorted_map:
                    max_fret = max(f[0] for f in sorted_map)
                    # Use logarithmic spacing (frets get closer together)
                    # Simplified: use square root spacing approximation
                    if max_fret > 0:
                        progress = fret_num / max_fret
                        fret_x = neck_left + int(neck_width * progress * 0.9)  # 0.9 accounts for taper
                    else:
                        fret_x = neck_left + 20
                else:
                    fret_x = neck_left + int(neck_width * fret_num / 12)
            else:
                fret_x = neck_left + int(neck_width * fret_num / 12)
    
    # Apply perspective correction
    fret_x, string_y = apply_perspective_correction(fret_x, string_y, neck_box, fret_map)
    
    return (int(fret_x), int(string_y))


def draw_finger_marker(frame, x, y, finger_num, note_name, is_correct=False, size=22):
    """
    Draw a single finger position marker
    
    Args:
        frame: OpenCV frame (BGR format)
        x, y: Position coordinates
        finger_num: 1-4 (Index, Middle, Ring, Pinky)
        note_name: Note name (C, D, E, etc.)
        is_correct: Whether finger is in correct position (for feedback)
        size: Radius of the circle
    """
    # Get color based on finger
    base_color = FINGER_COLORS.get(finger_num, (255, 255, 255))
    
    # Adjust color if correct (make brighter/green tint)
    if is_correct:
        color = (0, 255, 100)  # Bright green
    else:
        color = base_color
    
    # Draw outer circle (white border)
    cv2.circle(frame, (x, y), size + 2, (255, 255, 255), 2)
    
    # Draw main circle
    cv2.circle(frame, (x, y), size, color, -1)
    
    # Draw inner circle for depth
    cv2.circle(frame, (x, y), size - 4, (255, 255, 255), 1)
    
    # Draw finger number
    finger_text = str(finger_num) if finger_num > 0 else "T"
    text_size = cv2.getTextSize(finger_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    text_x = x - text_size[0] // 2
    text_y = y + text_size[1] // 2
    
    # Draw text with shadow for readability
    cv2.putText(frame, finger_text, (text_x + 1, text_y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
    cv2.putText(frame, finger_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Draw note name below
    if note_name:
        note_text = note_name
        note_size = cv2.getTextSize(note_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        note_x = x - note_size[0] // 2
        note_y = y + size + 20
        
        # Background rectangle for note name
        padding = 4
        cv2.rectangle(frame, 
                     (note_x - padding, note_y - note_size[1] - padding),
                     (note_x + note_size[0] + padding, note_y + padding),
                     (0, 0, 0), -1)
        
        # Draw note name
        cv2.putText(frame, note_text, (note_x, note_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def render_chord_overlay(frame, chord_info, neck_box, fret_map, string_positions, 
                         detected_fingers=None, show_strings=False):
    """
    Render chord overlay on frame
    
    Args:
        frame: OpenCV frame (BGR format)
        chord_info: Dict with 'name' and 'fingering' (list of (string, fret, finger_name))
        neck_box: Tuple (x1, y1, x2, y2) of neck bounding box
        fret_map: List of (fret_number, (x1, y1, x2, y2)) tuples
        string_positions: List of 6 Y-coordinates for strings
        detected_fingers: Optional list of detected finger positions for feedback
        show_strings: Whether to draw string lines
    
    Returns:
        Modified frame
    """
    if not chord_info or not chord_info.get('fingering'):
        return frame
    
    if neck_box is None or not string_positions:
        return frame
    
    # Draw string lines if requested
    if show_strings and string_positions:
        x1, y1, x2, y2 = neck_box
        for string_y in string_positions:
            cv2.line(frame, (x1, string_y), (x2, string_y), (100, 100, 100), 1)
    
    # Render each finger position
    fingering = chord_info['fingering']
    
    for string_num, fret_num, finger_name in fingering:
        # Convert finger name to number
        finger_num = FINGER_MAP.get(finger_name, 0)
        
        # Calculate note name
        note_name = get_note_name(string_num, fret_num)
        
        # Calculate pixel position
        pos = calculate_position_pixel(string_num, fret_num, neck_box, fret_map, string_positions)
        
        if pos is None:
            continue
        
        x, y = pos
        
        # Check if finger is in correct position (for feedback)
        is_correct = False
        if detected_fingers:
            for detected in detected_fingers:
                if (detected.get('string') == string_num and 
                    detected.get('fret') == fret_num and
                    detected.get('finger') == finger_name):
                    is_correct = True
                    break
        
        # Draw marker
        draw_finger_marker(frame, x, y, finger_num, note_name, is_correct)
    
    return frame


def draw_chord_info_overlay(frame, chord_name, overlay_enabled=True, x=20, y=50):
    """
    Draw chord information overlay on frame
    
    Args:
        frame: OpenCV frame
        chord_name: Name of current chord
        overlay_enabled: Whether overlay is currently shown
        x, y: Position for overlay text
    """
    h, w = frame.shape[:2]
    
    # Background rectangle
    cv2.rectangle(frame, (x - 10, y - 30), (x + 300, y + 40), (0, 0, 0), -1)
    cv2.rectangle(frame, (x - 10, y - 30), (x + 300, y + 40), (0, 255, 255), 2)
    
    # Chord name
    cv2.putText(frame, f"Chord: {chord_name}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Overlay status
    status = "ON" if overlay_enabled else "OFF"
    status_color = (0, 255, 0) if overlay_enabled else (0, 0, 255)
    cv2.putText(frame, f"Overlay: {status}", (x, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
