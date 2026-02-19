"""
String Position Refinement using Edge Detection
Refines calculated string positions by detecting actual string edges in the image
"""

import cv2
import numpy as np
from string_calibration import refine_strings_multi_method


def calculate_confidence_scores(frame, neck_box, initial_positions, refined_positions,
                                edge_matches_per_string=None):
    """
    Calculate confidence scores for each string position.
    
    Returns:
        List of confidence scores (0-1) for each string
    """
    if not initial_positions or not refined_positions or len(initial_positions) != 6:
        return [0.5] * 6
    
    confidences = []
    neck_height = neck_box[3] - neck_box[1] if neck_box else 100
    
    for i in range(6):
        initial_pos = initial_positions[i]
        refined_pos = refined_positions[i]
        
        if edge_matches_per_string and i < len(edge_matches_per_string):
            edge_confidence = min(edge_matches_per_string[i] / 3.0, 1.0)
        else:
            edge_confidence = 0.5
        
        position_diff = abs(refined_pos - initial_pos)
        stability = 1.0 - min(position_diff / (neck_height * 0.1), 1.0)
        
        if i > 0:
            spacing = abs(refined_pos - refined_positions[i - 1])
            expected_spacing = neck_height / 6.0
            consistency = 1.0 - min(abs(spacing - expected_spacing) / expected_spacing, 1.0)
        else:
            consistency = 0.7
        
        confidence = (edge_confidence * 0.5 + stability * 0.3 + consistency * 0.2)
        confidences.append(max(0.0, min(1.0, confidence)))
    
    return confidences


def refine_string_positions_with_edges(frame, neck_box, initial_positions,
                                       fret_map=None, threshold_ratio=0.12):
    """
    Refine string positions using multi-method approach.
    
    Returns:
        Tuple: (refined_positions, confidence_scores)
    """
    if neck_box is None or not initial_positions or len(initial_positions) != 6:
        return initial_positions, [0.5] * 6
    
    refined, edge_match_counts = refine_strings_multi_method(
        frame, neck_box, fret_map, None, initial_positions,
        use_fret_intersections=(fret_map is not None),
        use_edge_detection=True
    )
    
    # Clamp to playable region so refinement never returns out-of-bounds positions
    y_min, y_max = neck_box[1], neck_box[3]
    refined = [max(y_min, min(y_max, int(y))) for y in refined]
    
    confidences = calculate_confidence_scores(
        frame, neck_box, initial_positions, refined,
        edge_matches_per_string=edge_match_counts
    )
    
    return refined, confidences
