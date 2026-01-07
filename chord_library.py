"""
Chord Library - Common guitar chord fingerings
Format: {chord_name: [(string, fret, finger_name), ...]}
"""

CHORD_LIBRARY = {
    # Open chords (beginner friendly)
    "C Major": [
        (2, 1, "Index"),    # String 2, Fret 1
        (4, 2, "Middle"),   # String 4, Fret 2
        (5, 3, "Ring"),     # String 5, Fret 3
    ],
    
    "G Major": [
        (1, 3, "Pinky"),    # String 1, Fret 3
        (5, 2, "Index"),    # String 5, Fret 2
        (6, 3, "Middle"),   # String 6, Fret 3
    ],
    
    "D Major": [
        (1, 2, "Index"),    # String 1, Fret 2
        (2, 3, "Ring"),     # String 2, Fret 3
        (3, 2, "Middle"),   # String 3, Fret 2
    ],
    
    "E Minor": [
        (4, 2, "Index"),    # String 4, Fret 2
        (5, 2, "Middle"),   # String 5, Fret 2
    ],
    
    "A Minor": [
        (2, 1, "Index"),    # String 2, Fret 1
        (3, 2, "Ring"),     # String 3, Fret 2
        (4, 2, "Middle"),   # String 4, Fret 2
    ],
    
    "E Major": [
        (3, 1, "Index"),    # String 3, Fret 1
        (4, 2, "Ring"),     # String 4, Fret 2
        (5, 2, "Middle"),   # String 5, Fret 2
    ],
    
    "A Major": [
        (2, 2, "Index"),    # String 2, Fret 2
        (3, 2, "Middle"),   # String 3, Fret 2
        (4, 2, "Ring"),     # String 4, Fret 2
    ],
    
    "D Minor": [
        (1, 1, "Index"),    # String 1, Fret 1
        (2, 3, "Ring"),     # String 2, Fret 3
        (3, 2, "Middle"),   # String 3, Fret 2
    ],
}

# Beginner progression (easiest to hardest)
BEGINNER_CHORDS = [
    "E Minor",
    "A Minor", 
    "D Major",
    "G Major",
    "C Major",
    "E Major",
    "A Major",
    "D Minor",
]

def get_chord_info(chord_name):
    """Get chord fingering information"""
    if chord_name not in CHORD_LIBRARY:
        return None
    
    return {
        'name': chord_name,
        'fingering': CHORD_LIBRARY[chord_name],
        'finger_count': len(CHORD_LIBRARY[chord_name])
    }

def check_finger_position(detected_finger, target_position, tolerance=0.5):
    """
    Check if a detected finger matches a target position
    
    Args:
        detected_finger: dict with 'string', 'fret', 'finger'
        target_position: tuple (string, fret, finger_name)
        tolerance: fret tolerance (0.5 = within half a fret)
    
    Returns:
        bool: True if finger is in correct position
    """
    target_string, target_fret, target_finger = target_position
    
    # Check if it's the right finger
    if detected_finger['finger'] != target_finger:
        return False
    
    # Check if it's the right string
    if detected_finger['string'] != target_string:
        return False
    
    # Check if it's the right fret (with tolerance)
    fret_diff = abs(detected_finger['fret'] - target_fret)
    if fret_diff > tolerance:
        return False
    
    return True

def evaluate_chord(detected_fingers, chord_name):
    """
    Evaluate how well the detected fingers match the target chord
    
    Returns:
        dict: {
            'correct_fingers': [...],
            'missing_fingers': [...],
            'incorrect_fingers': [...],
            'accuracy': 0-100
        }
    """
    chord_info = get_chord_info(chord_name)
    if not chord_info:
        return None
    
    target_positions = chord_info['fingering']
    
    correct_fingers = []
    missing_fingers = []
    incorrect_fingers = list(detected_fingers)  # Start with all detected
    
    # Check each target position
    for target_pos in target_positions:
        found = False
        
        for detected in detected_fingers:
            if check_finger_position(detected, target_pos):
                correct_fingers.append({
                    'target': target_pos,
                    'detected': detected
                })
                if detected in incorrect_fingers:
                    incorrect_fingers.remove(detected)
                found = True
                break
        
        if not found:
            missing_fingers.append(target_pos)
    
    # Calculate accuracy
    total_required = len(target_positions)
    correct_count = len(correct_fingers)
    accuracy = (correct_count / total_required * 100) if total_required > 0 else 0
    
    return {
        'correct_fingers': correct_fingers,
        'missing_fingers': missing_fingers,
        'incorrect_fingers': incorrect_fingers,
        'accuracy': accuracy,
        'is_perfect': accuracy == 100 and len(incorrect_fingers) == 0
    }
