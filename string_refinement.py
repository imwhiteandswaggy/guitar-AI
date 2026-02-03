"""
String Position Refinement using Edge Detection
Refines calculated string positions by detecting actual string edges in the image
"""

import cv2
import numpy as np
from string_calibration import (
    refine_strings_with_fret_intersections,
    refine_strings_with_edge_detection_improved,
    refine_strings_multi_method
)


def calculate_confidence_scores(frame, neck_box, initial_positions, refined_positions, 
                                edge_matches_per_string=None):
    """
    Calculate confidence scores for each string position
    
    Args:
        frame: Input frame
        neck_box: Neck bounding box
        initial_positions: Initial calculated positions
        refined_positions: Refined positions from edge detection
        edge_matches_per_string: Optional list of edge match counts per string
    
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
        
        # Edge detection confidence
        if edge_matches_per_string and i < len(edge_matches_per_string):
            edge_confidence = min(edge_matches_per_string[i] / 3.0, 1.0)
        else:
            edge_confidence = 0.5
        
        # Position stability (how close refined is to initial)
        position_diff = abs(refined_pos - initial_pos)
        stability = 1.0 - min(position_diff / (neck_height * 0.1), 1.0)
        
        # Consistency (how close to expected spacing)
        if i > 0:
            spacing = abs(refined_pos - refined_positions[i-1])
            expected_spacing = neck_height / 6.0
            consistency = 1.0 - min(abs(spacing - expected_spacing) / expected_spacing, 1.0)
        else:
            consistency = 0.7
        
        # Combined confidence
        confidence = (edge_confidence * 0.5 + stability * 0.3 + consistency * 0.2)
        confidences.append(max(0.0, min(1.0, confidence)))
    
    return confidences


def refine_string_positions_with_edges(frame, neck_box, initial_positions, 
                                       fret_map=None, threshold_ratio=0.12):
    """
    Refine string positions using improved multi-method approach.
    Uses fret intersections and improved edge detection for better accuracy.
    
    Args:
        frame: Input BGR frame
        neck_box: Tuple (x1, y1, x2, y2) of neck bounding box
        initial_positions: List of 6 Y-coordinates (calculated string positions)
        fret_map: Optional fret map for intersection-based refinement
        threshold_ratio: Ratio of neck height to use as search threshold (default 0.12)
    
    Returns:
        Tuple: (refined_positions, confidence_scores)
    """
    if neck_box is None or not initial_positions or len(initial_positions) != 6:
        return initial_positions, [0.5] * 6
    
    # Use multi-method refinement
    refined, edge_match_counts = refine_strings_multi_method(
        frame, neck_box, fret_map, None, initial_positions,
        use_fret_intersections=(fret_map is not None),
        use_edge_detection=True
    )
    
    # Calculate confidence scores using edge match counts
    confidences = calculate_confidence_scores(
        frame, neck_box, initial_positions, refined, 
        edge_matches_per_string=edge_match_counts
    )
    
    return refined, confidences


def visualize_string_refinement(frame, neck_box, initial_positions, refined_positions):
    """
    Draw visualization of initial vs refined string positions for debugging.
    
    Args:
        frame: Input BGR frame (will be modified)
        neck_box: Tuple (x1, y1, x2, y2) of neck bounding box
        initial_positions: List of initial Y-coordinates (blue)
        refined_positions: List of refined Y-coordinates (green)
    """
    if neck_box is None:
        return
    
    x1, y1, x2, y2 = neck_box
    w = x2 - x1
    
    # Draw initial positions in blue
    for y in initial_positions:
        cv2.line(frame, (x1, y), (x2, y), (255, 0, 0), 1)  # Blue
    
    # Draw refined positions in green
    for y in refined_positions:
        cv2.line(frame, (x1, y), (x2, y), (0, 255, 0), 2)  # Green
