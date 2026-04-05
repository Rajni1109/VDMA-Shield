import cv2
import time
import numpy as np
from vision_engine import VisionEngine
from motion_engine import MotionAnalyzer
from depth_engine import DepthEstimator
from behavior_engine import BehaviorEngine
from utils import Visualizer
from evidence_manager import VideoEvidenceManager
from behavior_accumulator import BehaviorAccumulator

def main():
    video_path = 'snatch3.mp4'
    cap = cv2.VideoCapture(video_path)
    
    # --- 1. VIDEO WRITER INITIALIZATION ---
    # Get original video properties for consistency
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0: original_fps = 20 # Fallback
    
    # Define codec and create VideoWriter object
    output_filename = f"processed_output_2{int(time.time())}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    out_writer = cv2.VideoWriter(output_filename, fourcc, original_fps, (frame_width, frame_height))
    # ----------------------------------------

    vision, motion, depth_eng = VisionEngine(), MotionAnalyzer(), DepthEstimator()
    behave, viz = BehaviorEngine(), Visualizer()
    evidence_mgr = VideoEvidenceManager(buffer_sec=10, fps=20)
    accumulator = BehaviorAccumulator(window_sec=10, count_threshold=3)

    prev_time = 0
    frame_count = 0
    show_depth_only = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. SAVE UNBLURRED FRAME TO EVIDENCE BUFFER
        evidence_mgr.add_frame(frame)              

        # 2. ANALYSIS LOGIC
        if frame_count % 3 == 0:
            results = vision.detect_and_track(frame)
            motion_map = motion.get_flow_magnitude(frame)
            depth_map = depth_eng.get_depth_map(frame)
            alerts = behave.analyze(results, motion_map, depth_map)
            
            last_results, last_alerts, last_depth = results, alerts, depth_map
        else:
            results, alerts, depth_map = last_results, last_alerts, last_depth
        
        frame_count += 1
        
        # 3. EVIDENCE TRIGGER
        for alert in alerts:
            if accumulator.add_and_check(alert):
                evidence_mgr.save_evidence(alert)

        # 4. DISPLAY & SAVE LOGIC
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        if show_depth_only and depth_map is not None:
            depth_vis = (depth_map * 255).astype(np.uint8) if depth_map.max() <= 1.0 else depth_map.astype(np.uint8)
            output = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
        else:
            output = viz.draw(frame.copy(), results, alerts, fps) 
        
        # --- 2. WRITE FRAME TO VIDEO FILE ---
        out_writer.write(output)
        # ------------------------------------
        
        cv2.imshow("VDM Analyzer - Surveillance", output)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('d'): show_depth_only = not show_depth_only

    # --- 3. RELEASE EVERYTHING ---
    cap.release()
    out_writer.release() # Very important to finalize the file
    cv2.destroyAllWindows()
    print(f"Inference video saved as: {output_filename}")

if __name__ == "__main__":
    main()