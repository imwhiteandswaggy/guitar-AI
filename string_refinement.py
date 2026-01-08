"""
String Position Refinement using Edge Detection
Refines calculated string positions by detecting actual string edges in the image
"""

import cv2
import numpy as np


def refine_string_positions_with_edges(frame, neck_box, initial_positions, threshold_ratio=0.15):
    """
    Refine string positions using edge detection to find actual string locations.
    
    Args:
        frame: Input BGR frame
        neck_box: Tuple (x1, y1, x2, y2) of neck bounding box
        initial_positions: List of 6 Y-coordinates (calculated string positions)
        threshold_ratio: Ratio of neck height to use as search threshold (default 0.15)
    
    Returns:
        List of refined Y-coordinates (same length as initial_positions)
    """
    if neck_box is None or not initial_positions or len(initial_positions) != 6:
        return initial_positions
    
    x1, y1, x2, y2 = neck_box
    
    # Extract neck ROI
    neck_roi = frame[y1:y2, x1:x2]
    if neck_roi.size == 0:
        return initial_positions
    
    # Convert to grayscale
    gray = cv2.cvtColor(neck_roi, cv2.COLOR_BGR2GRAY)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # HoughLinesP for horizontal lines (strings)
    # Look for lines that are mostly horizontal (strings)
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=30,
        minLineLength=int((x2 - x1) * 0.3),  # At least 30% of neck width
        maxLineGap=10
    )
    
    if lines is None or len(lines) == 0:
        return initial_positions
    
    # Calculate threshold for matching lines to initial positions
    neck_height = y2 - y1
    search_threshold = int(neck_height * threshold_ratio)
    
    # Filter horizontal lines and refine positions
    refined_positions = []
    
    for initial_y in initial_positions:
        # Find lines near this position
        nearby_line_ys = []
        
        for line in lines:
            x1_line, y1_line, x2_line, y2_line = line[0]
            
            # Check if line is mostly horizontal (within 5 pixels vertical difference)
            if abs(y1_line - y2_line) < 5:
                # Average Y position of the line
                line_y_roi = (y1_line + y2_line) // 2
                
                # Convert to full frame coordinates
                line_y_full = y1 + line_y_roi
                
                # Check if this line is near our initial position
                if abs(line_y_full - initial_y) < search_threshold:
                    nearby_line_ys.append(line_y_full)
        
        # Use average of nearby lines, or keep original if no matches
        if nearby_line_ys:
            refined_y = int(np.mean(nearby_line_ys))
            refined_positions.append(refined_y)
        else:
            refined_positions.append(initial_y)
    
    return refined_positions


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
