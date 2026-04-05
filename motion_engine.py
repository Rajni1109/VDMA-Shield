import cv2
import numpy as np

class MotionAnalyzer:
    def __init__(self):
        self.prev_gray = None

    def get_flow_magnitude(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros_like(gray)
        
        # Farneback is robust for dense flow
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev_gray = gray
        
        # Convert to magnitude for visualization/logic
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)