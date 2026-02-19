"""
String Position Tracking with Temporal Smoothing
Implements EMA and median filtering for stable string positions
"""

import numpy as np
from collections import deque


class StringPositionTracker:
    """
    Tracks string positions over time using exponential moving average (EMA).
    Prevents glitchy jumps by smoothing position updates.
    """
    
    def __init__(self, alpha=0.3, max_change_threshold=20):
        self.alpha = alpha
        self.max_change_threshold = max_change_threshold
        self.smoothed = [None] * 6
        self.frame_count = 0
        
    def update(self, new_positions):
        if not new_positions or len(new_positions) != 6:
            return self.smoothed if self.smoothed[0] is not None else new_positions
        
        if self.smoothed[0] is None:
            self.smoothed = list(new_positions)
            self.frame_count = 1
            return self.smoothed
        
        updated_positions = []
        for i in range(6):
            old_pos = self.smoothed[i]
            new_pos = new_positions[i]
            change = abs(new_pos - old_pos)
            
            if change <= self.max_change_threshold:
                smoothed_pos = self.alpha * new_pos + (1 - self.alpha) * old_pos
                updated_positions.append(int(smoothed_pos))
            else:
                updated_positions.append(old_pos)
        
        self.smoothed = updated_positions
        self.frame_count += 1
        return self.smoothed
    
    def reset(self):
        self.smoothed = [None] * 6
        self.frame_count = 0
    
    def get_positions(self):
        return self.smoothed if self.smoothed[0] is not None else None


class MedianFilterTracker:
    """Uses median filtering over a window of frames to remove outliers."""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = [deque(maxlen=window_size) for _ in range(6)]
    
    def update(self, new_positions):
        if not new_positions or len(new_positions) != 6:
            return new_positions
        
        filtered_positions = []
        for i in range(6):
            self.history[i].append(new_positions[i])
            if len(self.history[i]) >= 3:
                median_pos = int(np.median(list(self.history[i])))
            else:
                median_pos = new_positions[i]
            filtered_positions.append(median_pos)
        
        return filtered_positions
    
    def reset(self):
        self.history = [deque(maxlen=self.window_size) for _ in range(6)]


class HybridTracker:
    """
    Combines EMA + Median filtering + Outlier rejection
    for best string tracking results.
    """
    
    def __init__(self, alpha=0.3, window_size=5, max_change=20):
        self.ema_tracker = StringPositionTracker(alpha=alpha, max_change_threshold=max_change)
        self.median_tracker = MedianFilterTracker(window_size=window_size)
        self.confidence_threshold = 0.5
    
    def update(self, new_positions, confidence_scores=None, y_min=None, y_max=None):
        if not new_positions or len(new_positions) != 6:
            return self.ema_tracker.get_positions() or new_positions
        
        # Step 1: Median filter to remove outliers
        median_filtered = self.median_tracker.update(new_positions)
        
        # Step 2: Confidence-based filtering
        if confidence_scores:
            for i in range(6):
                if confidence_scores[i] < self.confidence_threshold:
                    prev_pos = self.ema_tracker.get_positions()
                    if prev_pos:
                        median_filtered[i] = prev_pos[i]
        
        # Clamp to playable range so tracker state never drifts outside neck
        if y_min is not None and y_max is not None:
            median_filtered = [max(y_min, min(y_max, int(y))) for y in median_filtered]
        
        # Step 3: EMA smoothing
        smoothed = self.ema_tracker.update(median_filtered)
        if y_min is not None and y_max is not None:
            smoothed = [max(y_min, min(y_max, int(y))) for y in smoothed]
        return smoothed
    
    def get_positions(self):
        return self.ema_tracker.get_positions()
    
    def reset(self):
        self.ema_tracker.reset()
        self.median_tracker.reset()
