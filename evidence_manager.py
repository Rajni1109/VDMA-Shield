import cv2
from collections import deque
from datetime import datetime
import os

class VideoEvidenceManager:
    def __init__(self, buffer_sec=10, fps=20, output_dir="evidence"):
        self.fps = fps
        self.buffer_size = int(buffer_sec * fps)
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def add_frame(self, frame):
        """Adds current frame to the rolling buffer."""
        self.frame_buffer.append(frame.copy())

    def save_evidence(self, event_name):
        """Saves the contents of the buffer to a file."""
        if not self.frame_buffer: return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize event name for filename
        clean_name = event_name.replace(':', '').replace(' ', '_')
        filename = os.path.join(self.output_dir, f"{clean_name}_{timestamp}.mp4")
        
        h, w = self.frame_buffer[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, self.fps, (w, h))
        
        for f in self.frame_buffer:
            out.write(f)
        out.release()
        print(f"--- [EVIDENCE SAVED]: {filename} ---")