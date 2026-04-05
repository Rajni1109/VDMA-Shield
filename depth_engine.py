import torch
import cv2
import numpy as np

class DepthEstimator:
    def __init__(self):
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(self.device).eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    def get_depth_map(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=frame.shape[:2], mode="bicubic").squeeze()
        return cv2.normalize(prediction.cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)