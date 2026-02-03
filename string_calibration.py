"""
String Position Calibration and Refinement
Multiple methods to improve string position accuracy
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def refine_strings_with_fret_intersections(neck_box, fret_map, string_positions, frame=None):
    """
    Refine string positions using fret intersections.
    Strings should cross frets at consistent points - use this to refine positions.
    
    Args:
        neck_box: Tuple (x1, y1, x2, y2) of neck bounding box
        fret_map: List of (fret_number, (x1, y1, x2, y2)) tuples
        string_positions: Initial string Y positions
        frame: Optional frame for visualization
    
    Returns:
        Tuple: (refined_string_positions, edge_match_counts)
    """
    if not fret_map or not string_positions or len(string_positions) != 6:
        return string_positions, [0] * 6
    
    if neck_box is None:
        return string_positions, [0] * 6
    
    # Sort frets by number
    sorted_frets = sorted(fret_map, key=lambda x: x[0])
    
    # For each string, find where it should intersect frets
    # Strings should form straight lines across frets
    refined_positions = []
    edge_match_counts = []
    
    for string_idx, string_y in enumerate(string_positions):
        # Collect intersection points with frets
        intersection_points = []
        
        for fret_num, fret_box in sorted_frets:
            # Fret box: (x1, y1, x2, y2)
            fret_y1 = fret_box[1]
            fret_y2 = fret_box[3]
            
            # String should pass through the fret at string_y
            # But if fret is tilted, we need to interpolate
            fret_center_x = (fret_box[0] + fret_box[2]) // 2
            
            # Check if string_y is within fret bounds
            if fret_y1 <= string_y <= fret_y2:
                intersection_points.append((fret_center_x, string_y))
        
        # If we have multiple intersection points, fit a line
        if len(intersection_points) >= 2:
            # Fit line through intersections
            points = np.array(intersection_points)
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            
            # Fit line: y = mx + b
            # But strings are mostly horizontal, so average Y
            avg_y = np.mean(y_coords)
            
            # Use average Y, but weight by number of intersections
            refined_y = int(avg_y)
            refined_positions.append(refined_y)
            edge_match_counts.append(len(intersection_points))
        else:
            # Not enough intersections, keep original
            refined_positions.append(string_y)
            edge_match_counts.append(0)
    
    return refined_positions, edge_match_counts


def calculate_adaptive_thresholds(gray_roi):
    """
    Calculate adaptive thresholds based on image characteristics
    
    Args:
        gray_roi: Grayscale ROI of neck area
    
    Returns:
        Tuple (canny_low, canny_high, hough_threshold)
    """
    # Calculate image statistics
    mean_intensity = np.mean(gray_roi)
    std_intensity = np.std(gray_roi)
    
    # Adjust Canny thresholds based on lighting
    # Brighter images need higher thresholds
    base_low = 30
    base_high = 100
    
    # Scale based on intensity
    intensity_factor = mean_intensity / 128.0  # Normalize to 128
    contrast_factor = std_intensity / 50.0  # Normalize to 50
    
    canny_low = int(base_low * intensity_factor * (1 + contrast_factor * 0.5))
    canny_high = int(base_high * intensity_factor * (1 + contrast_factor * 0.5))
    
    # Clamp to reasonable values
    canny_low = max(20, min(60, canny_low))
    canny_high = max(60, min(200, canny_high))
    
    # Hough threshold based on image size and contrast
    roi_height = gray_roi.shape[0]
    hough_threshold = max(15, int(roi_height * 0.05 * (1 + contrast_factor)))
    
    return canny_low, canny_high, hough_threshold


def refine_strings_with_edge_detection_improved(frame, neck_box, initial_positions, 
                                                 fret_map=None, threshold_ratio=0.12):
    """
    Improved edge detection with adaptive thresholds and better filtering.
    
    Args:
        frame: Input BGR frame
        neck_box: Tuple (x1, y1, x2, y2) of neck bounding box
        initial_positions: List of 6 Y-coordinates
        fret_map: Optional fret map for filtering
        threshold_ratio: Search threshold ratio
    
    Returns:
        Refined string positions
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
    
    # Calculate adaptive thresholds based on image quality
    canny_low, canny_high, hough_threshold = calculate_adaptive_thresholds(gray)
    
    # Apply adaptive thresholding for better string detection
    # This helps with varying lighting conditions
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Also try Canny with adaptive thresholds
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    
    # Combine both methods
    combined = cv2.bitwise_or(adaptive, edges)
    
    # Morphological operations to connect string segments
    kernel = np.ones((3, 1), np.uint8)  # Horizontal kernel
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    # HoughLinesP with adaptive parameters
    neck_width = x2 - x1
    lines = cv2.HoughLinesP(
        combined,
        rho=1,
        theta=np.pi/180,
        threshold=hough_threshold,  # Use adaptive threshold
        minLineLength=int(neck_width * 0.4),  # At least 40% of neck width
        maxLineGap=5  # Smaller gap for better continuity
    )
    
    if lines is None or len(lines) == 0:
        return initial_positions, [0] * 6
    
    # Filter and group horizontal lines
    neck_height = y2 - y1
    search_threshold = int(neck_height * threshold_ratio)
    
    # Group lines by Y position
    line_groups = {}
    for line in lines:
        x1_line, y1_line, x2_line, y2_line = line[0]
        
        # Check if horizontal (within 3 pixels)
        if abs(y1_line - y2_line) < 3:
            line_y_roi = (y1_line + y2_line) // 2
            line_y_full = y1 + line_y_roi
            
            # Group nearby lines
            grouped = False
            for group_y in line_groups.keys():
                if abs(line_y_full - group_y) < search_threshold:
                    line_groups[group_y].append(line_y_full)
                    grouped = True
                    break
            
            if not grouped:
                line_groups[line_y_full] = [line_y_full]
    
    # Refine each string position and track edge matches
    refined_positions = []
    edge_match_counts = []
    
    for initial_y in initial_positions:
        # Find closest line group
        best_match = None
        min_dist = float('inf')
        match_count = 0
        
        for group_y, group_lines in line_groups.items():
            dist = abs(group_y - initial_y)
            if dist < min_dist and dist < search_threshold:
                min_dist = dist
                best_match = group_y
                match_count = len(group_lines)
        
        if best_match is not None:
            # Use average of group
            refined_y = int(np.mean(line_groups[best_match]))
            refined_positions.append(refined_y)
            edge_match_counts.append(match_count)
        else:
            refined_positions.append(initial_y)
            edge_match_counts.append(0)
    
    return refined_positions, edge_match_counts


def manual_string_calibration(frame, neck_box):
    """
    Interactive manual calibration - user clicks on each string.
    
    Args:
        frame: Current frame
        neck_box: Neck bounding box
    
    Returns:
        List of 6 Y-coordinates (string positions) or None if cancelled
    """
    if neck_box is None:
        return None
    
    x1, y1, x2, y2 = neck_box
    calibration_frame = frame.copy()
    
    # Draw neck box
    cv2.rectangle(calibration_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Instructions
    cv2.putText(calibration_frame, "Click on each string (1-6, top to bottom)", 
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(calibration_frame, "Press ESC to cancel, ENTER when done", 
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    string_positions = []
    click_count = 0
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal click_count, string_positions, calibration_frame
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is within neck
            if x1 <= x <= x2 and y1 <= y <= y2:
                string_positions.append(y)
                click_count += 1
                
                # Draw marker
                cv2.circle(calibration_frame, (x, y), 5, (0, 255, 255), -1)
                cv2.putText(calibration_frame, str(click_count), (x + 10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                if click_count >= 6:
                    cv2.putText(calibration_frame, "All strings marked! Press ENTER", 
                               (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.namedWindow('String Calibration')
    cv2.setMouseCallback('String Calibration', mouse_callback)
    
    while True:
        cv2.imshow('String Calibration', calibration_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            cv2.destroyWindow('String Calibration')
            return None
        elif key == 13 and click_count >= 6:  # ENTER
            cv2.destroyWindow('String Calibration')
            # Sort by Y position (top to bottom)
            string_positions.sort()
            return string_positions[:6]
    
    return None


def refine_strings_multi_method(frame, neck_box, fret_map, nut_box, 
                                 initial_positions, use_fret_intersections=True,
                                 use_edge_detection=True):
    """
    Apply multiple refinement methods in sequence for best accuracy.
    
    Args:
        frame: Input frame
        neck_box: Neck bounding box
        fret_map: Fret map
        nut_box: Nut box
        initial_positions: Initial string positions
        use_fret_intersections: Use fret intersection method
        use_edge_detection: Use edge detection method
    
    Returns:
        Refined string positions
    """
    refined = initial_positions.copy()
    
    # Method 1: Fret intersections (if frets detected)
    if use_fret_intersections and fret_map:
        refined, fret_edge_counts = refine_strings_with_fret_intersections(
            neck_box, fret_map, refined, frame
        )
        # Combine edge counts (fret intersections count as matches)
        for i in range(6):
            edge_match_counts[i] += fret_edge_counts[i]
    
    # Method 2: Improved edge detection
    if use_edge_detection and frame is not None:
        refined = refine_strings_with_edge_detection_improved(
            frame, neck_box, refined, fret_map
        )
    
    return refined
