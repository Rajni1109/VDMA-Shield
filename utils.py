import cv2
from datetime import datetime
import numpy as np

class Visualizer:
    def draw(self, frame, results_dict, alerts, fps):                
        pose_results = results_dict.get('pose')
        det_results = results_dict.get('detection')

        # --- UPDATE: BLUR FACES (Privacy Masking) ---
        if pose_results and pose_results.keypoints is not None:
            # keypoints.xy shape: [num_people, 17, 2]
            kpts_all = pose_results.keypoints.xy.cpu().numpy()
            
            for kpts in kpts_all:
                # Keypoint 0 = Nose (best for face center)
                nose = kpts[0]
                nx, ny = int(nose[0]), int(nose[1])
                
                # Check if nose is detected (not 0,0)
                if nx > 0 and ny > 0:
                    # Define face region (adjust radius as needed)
                    r = 45 
                    h, w = frame.shape[:2]
                    y1, y2 = max(0, ny - r), min(h, ny + r)
                    x1, x2 = max(0, nx - r), min(w, nx + r)
                    
                    face_roi = frame[y1:y2, x1:x2]
                    if face_roi.size > 0:
                        # Heavy Gaussian blur for privacy
                        blurred = cv2.GaussianBlur(face_roi, (51, 51), 30)
                        frame[y1:y2, x1:x2] = blurred
        # ---------------------------------------------

        # Draw Skeletons
        if pose_results:
            frame = pose_results.plot(img=frame, kpt_radius=3, labels=True)
            
        # Draw Objects
        if det_results:
            frame = det_results.plot(img=frame, boxes=True, labels=True)

        # Draw Telemetry (Top Right Corner)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame, timestamp, (frame.shape[1] - 320, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        # Draw Alerts
        for i, text in enumerate(alerts):
            cv2.putText(frame, text, (20, 40 + (i * 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        return frame