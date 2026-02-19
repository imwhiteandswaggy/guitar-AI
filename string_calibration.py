"""
String Position Calibration and Refinement
Multiple methods to improve string position accuracy
"""

import cv2
import numpy as np


def refine_strings_with_fret_intersections(neck_box, fret_map, string_positions, frame=None):
    """
    Refine string positions using fret intersections.
    Clamps refined Y values to neck vertical range so strings stay on the fretboard.
    
    Returns:
        Tuple: (refined_string_positions, edge_match_counts)
    """
    if not fret_map or not string_positions or len(string_positions) != 6:
        return string_positions, [0] * 6
    if neck_box is None:
        return string_positions, [0] * 6
    
    y_min, y_max = neck_box[1], neck_box[3]
    sorted_frets = sorted(fret_map, key=lambda x: x[0])
    refined_positions = []
    edge_match_counts = []
    
    for string_idx, string_y in enumerate(string_positions):
        intersection_points = []
        
        for fret_num, fret_box in sorted_frets:
            fret_y1 = fret_box[1]
            fret_y2 = fret_box[3]
            fret_center_x = (fret_box[0] + fret_box[2]) // 2
            
            if fret_y1 <= string_y <= fret_y2:
                intersection_points.append((fret_center_x, string_y))
        
        if len(intersection_points) >= 2:
            points = np.array(intersection_points)
            avg_y = np.mean(points[:, 1])
            refined_y = int(avg_y)
            refined_y = max(y_min, min(y_max, refined_y))
            refined_positions.append(refined_y)
            edge_match_counts.append(len(intersection_points))
        else:
            clamped_y = max(y_min, min(y_max, string_y))
            refined_positions.append(clamped_y)
            edge_match_counts.append(0)
    
    return refined_positions, edge_match_counts


def calculate_adaptive_thresholds(gray_roi):
    """Calculate adaptive thresholds based on image characteristics."""
    mean_intensity = np.mean(gray_roi)
    std_intensity = np.std(gray_roi)
    
    intensity_factor = mean_intensity / 128.0
    contrast_factor = std_intensity / 50.0
    
    canny_low = int(30 * intensity_factor * (1 + contrast_factor * 0.5))
    canny_high = int(100 * intensity_factor * (1 + contrast_factor * 0.5))
    
    canny_low = max(20, min(60, canny_low))
    canny_high = max(60, min(200, canny_high))
    
    roi_height = gray_roi.shape[0]
    hough_threshold = max(15, int(roi_height * 0.05 * (1 + contrast_factor)))
    
    return canny_low, canny_high, hough_threshold


def refine_strings_with_edge_detection_improved(frame, neck_box, initial_positions,
                                                 fret_map=None, threshold_ratio=0.12):
    """
    Improved edge detection with adaptive thresholds.
    
    Returns:
        Tuple: (refined_positions, edge_match_counts)
    """
    if neck_box is None or not initial_positions or len(initial_positions) != 6:
        return initial_positions, [0] * 6
    
    x1, y1, x2, y2 = neck_box
    neck_roi = frame[y1:y2, x1:x2]
    if neck_roi.size == 0:
        return initial_positions, [0] * 6
    
    gray = cv2.cvtColor(neck_roi, cv2.COLOR_BGR2GRAY)
    canny_low, canny_high, hough_threshold = calculate_adaptive_thresholds(gray)
    
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    combined = cv2.bitwise_or(adaptive, edges)
    
    kernel = np.ones((3, 1), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    neck_width = x2 - x1
    lines = cv2.HoughLinesP(
        combined, rho=1, theta=np.pi / 180, threshold=hough_threshold,
        minLineLength=int(neck_width * 0.4), maxLineGap=5
    )
    
    if lines is None or len(lines) == 0:
        return initial_positions, [0] * 6
    
    neck_height = y2 - y1
    search_threshold = int(neck_height * threshold_ratio)
    
    line_groups = {}
    for line in lines:
        x1_line, y1_line, x2_line, y2_line = line[0]
        if abs(y1_line - y2_line) < 3:
            line_y_roi = (y1_line + y2_line) // 2
            line_y_full = y1 + line_y_roi
            # Only keep lines inside neck vertical range
            if line_y_full < y1 or line_y_full > y2:
                continue
            
            grouped = False
            for group_y in line_groups.keys():
                if abs(line_y_full - group_y) < search_threshold:
                    line_groups[group_y].append(line_y_full)
                    grouped = True
                    break
            if not grouped:
                line_groups[line_y_full] = [line_y_full]
    
    refined_positions = []
    edge_match_counts = []
    
    for initial_y in initial_positions:
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
            refined_y = int(np.mean(line_groups[best_match]))
            refined_y = max(y1, min(y2, refined_y))
            refined_positions.append(refined_y)
            edge_match_counts.append(match_count)
        else:
            clamped_y = max(y1, min(y2, initial_y))
            refined_positions.append(clamped_y)
            edge_match_counts.append(0)
    
    return refined_positions, edge_match_counts


def refine_strings_multi_method(frame, neck_box, fret_map, nut_box,
                                 initial_positions, use_fret_intersections=True,
                                 use_edge_detection=True):
    """
    Apply multiple refinement methods in sequence for best accuracy.
    
    Returns:
        Tuple: (refined_positions, edge_match_counts)
    """
    refined = list(initial_positions)
    edge_match_counts = [0] * 6
    
    # Method 1: Fret intersections
    if use_fret_intersections and fret_map:
        refined, fret_edge_counts = refine_strings_with_fret_intersections(
            neck_box, fret_map, refined, frame
        )
        for i in range(6):
            edge_match_counts[i] += fret_edge_counts[i]
    
    # Method 2: Improved edge detection
    if use_edge_detection and frame is not None:
        refined, edge_counts = refine_strings_with_edge_detection_improved(
            frame, neck_box, refined, fret_map
        )
        for i in range(6):
            edge_match_counts[i] += edge_counts[i]
    
    return refined, edge_match_counts
