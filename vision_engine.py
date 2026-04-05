from ultralytics import YOLO

class VisionEngine:
    def __init__(self, pose_model_path='yolo11n-pose.pt'):
        # 1. Load the model. DON'T call .half() here.
        # Moving to 'mps' is fine, but .half() causes the fuse error.
        self.pose_model = YOLO(pose_model_path).to('mps')
        
    def detect_and_track(self, frame):
        # 2. Let the track function handle the half-precision (FP16)
        # This prevents the mat1 and mat2 dtype mismatch.
        pose_results = self.pose_model.track(
            frame, 
            device='mps',
            imgsz=320,      # Keeps it fast
            half=True,       # <--- Turn on FP16 safely here
            persist=True, 
            classes=[0], 
            tracker="bytetrack.yaml", 
            verbose=False
        )
        
        # We return detection as None since we commented it out
        return {"pose": pose_results[0], "detection": None}