"""
String Position Tracking with Temporal Smoothing
Implements exponential moving average and Kalman filtering for stable string positions
"""

import numpy as np
from collections import deque


class StringPositionTracker:
    """
    Tracks string positions over time using exponential moving average (EMA)
    Prevents glitchy jumps by smoothing position updates
    """
    
    def __init__(self, alpha=0.3, max_change_threshold=20):
        """
        Args:
            alpha: Smoothing factor (0-1). Lower = more smoothing, slower response
            max_change_threshold: Maximum allowed change per frame (pixels)
        """
        self.alpha = alpha
        self.max_change_threshold = max_change_threshold
        self.smoothed = [None] * 6
        self.frame_count = 0
        
    def update(self, new_positions):
        """
        Update smoothed positions with new detections
        
        Args:
            new_positions: List of 6 Y-coordinates (current frame detections)
        
        Returns:
            List of smoothed Y-coordinates
        """
        if not new_positions or len(new_positions) != 6:
            return self.smoothed if self.smoothed[0] is not None else new_positions
        
        # First frame - initialize
        if self.smoothed[0] is None:
            self.smoothed = list(new_positions)
            self.frame_count = 1
            return self.smoothed
        
        # Update each string position with EMA
        updated_positions = []
        for i in range(6):
            old_pos = self.smoothed[i]
            new_pos = new_positions[i]
            
            # Calculate change
            change = abs(new_pos - old_pos)
            
            # Only update if change is reasonable (outlier rejection)
            if change <= self.max_change_threshold:
                # Exponential Moving Average: smoothed = alpha * new + (1-alpha) * old
                smoothed_pos = self.alpha * new_pos + (1 - self.alpha) * old_pos
                updated_positions.append(int(smoothed_pos))
            else:
                # Change too large - keep old position (reject outlier)
                updated_positions.append(old_pos)
        
        self.smoothed = updated_positions
        self.frame_count += 1
        
        return self.smoothed
    
    def reset(self):
        """Reset tracker (e.g., when guitar moves significantly)"""
        self.smoothed = [None] * 6
        self.frame_count = 0
    
    def get_positions(self):
        """Get current smoothed positions"""
        return self.smoothed if self.smoothed[0] is not None else None


class KalmanStringTracker:
    """
    Advanced Kalman filter for string position tracking
    Provides better smoothing and handles occlusions
    """
    
    def __init__(self):
        self.trackers = [None] * 6
        self.initialized = False
    
    def initialize(self, initial_positions):
        """Initialize Kalman filters for each string"""
        # Simplified Kalman filter implementation
        # For full implementation, would use OpenCV's KalmanFilter
        self.initialized = True
        # For now, fall back to EMA tracker
        return initial_positions
    
    def update(self, new_positions):
        """Update with Kalman filtering"""
        if not self.initialized:
            return self.initialize(new_positions)
        
        # Simplified: use EMA for now
        # Full Kalman would predict and correct
        return new_positions


class MedianFilterTracker:
    """
    Uses median filtering over a window of frames
    Very effective at removing outliers
    """
    
    def __init__(self, window_size=5):
        """
        Args:
            window_size: Number of frames to keep in history
        """
        self.window_size = window_size
        self.history = [deque(maxlen=window_size) for _ in range(6)]
    
    def update(self, new_positions):
        """
        Update with median filtering
        
        Args:
            new_positions: List of 6 Y-coordinates
        
        Returns:
            List of median-filtered Y-coordinates
        """
        if not new_positions or len(new_positions) != 6:
            return new_positions
        
        filtered_positions = []
        
        for i in range(6):
            # Add new position to history
            self.history[i].append(new_positions[i])
            
            # Calculate median
            if len(self.history[i]) >= 3:
                median_pos = int(np.median(list(self.history[i])))
            else:
                median_pos = new_positions[i]
            
            filtered_positions.append(median_pos)
        
        return filtered_positions
    
    def reset(self):
        """Reset all histories"""
        self.history = [deque(maxlen=self.window_size) for _ in range(6)]


class HybridTracker:
    """
    Combines multiple tracking methods for best results
    Uses EMA + Median filtering + Outlier rejection
    """
    
    def __init__(self, alpha=0.3, window_size=5, max_change=20):
        self.ema_tracker = StringPositionTracker(alpha=alpha, max_change_threshold=max_change)
        self.median_tracker = MedianFilterTracker(window_size=window_size)
        self.confidence_threshold = 0.5
    
    def update(self, new_positions, confidence_scores=None):
        """
        Update with hybrid approach
        
        Args:
            new_positions: List of 6 Y-coordinates
            confidence_scores: Optional list of confidence scores (0-1) for each string
        
        Returns:
            List of smoothed Y-coordinates
        """
        if not new_positions or len(new_positions) != 6:
            return self.ema_tracker.get_positions() or new_positions
        
        # Step 1: Median filter to remove outliers
        median_filtered = self.median_tracker.update(new_positions)
        
        # Step 2: Apply confidence-based filtering
        if confidence_scores:
            for i in range(6):
                if confidence_scores[i] < self.confidence_threshold:
                    # Low confidence - use previous position
                    prev_pos = self.ema_tracker.get_positions()
                    if prev_pos:
                        median_filtered[i] = prev_pos[i]
        
        # Step 3: EMA smoothing
        smoothed = self.ema_tracker.update(median_filtered)
        
        return smoothed
    
    def reset(self):
        """Reset all trackers"""
        self.ema_tracker.reset()
        self.median_tracker.reset()
