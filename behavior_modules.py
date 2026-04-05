import numpy as np
import time
from collections import defaultdict, deque

class ActionBuffer:
    def __init__(self, buffer_size=15):
        self.history = defaultdict(lambda: deque(maxlen=buffer_size))

    def update(self, obj_id, center, depth_val):
        self.history[obj_id].append({'pos': center, 'depth': depth_val, 'time': time.time()})

    def get_velocity_stats(self, obj_id):
        hist = self.history[obj_id]
        if len(hist) < 5: return 0, 0
        d_pos = np.linalg.norm(hist[-1]['pos'] - hist[0]['pos'])
        dt = hist[-1]['time'] - hist[0]['time']
        return (d_pos / dt if dt > 0 else 0), dt

class ViolenceDetector:
    def __init__(self, motion_thresh=60, dist_thresh=100, velocity_thresh=450, depth_thresh=0.05):
        self.motion_thresh = motion_thresh
        self.dist_thresh = dist_thresh
        self.velocity_thresh = velocity_thresh 
        self.depth_thresh = depth_thresh  
        self.buffer = ActionBuffer()

    def check(self, data, motion_map, depth_map):
        # Prevent overflow by normalizing depth to 0.0 - 1.0 range
        if depth_map.max() > 1.0:
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-5)
            
        people = data["people"]
        alerts = []
        h, w = motion_map.shape[:2]
        
        for i, p1 in enumerate(people):
            p1_center = np.array([(p1["box"][0]+p1["box"][2])/2, (p1["box"][1]+p1["box"][3])/2])
            cx1, cy1 = int(np.clip(p1_center[0], 0, w-1)), int(np.clip(p1_center[1], 0, h-1))
            p1_depth = depth_map[cy1, cx1]
            
            self.buffer.update(p1["id"], p1_center, p1_depth)
            v1_pixel, _ = self.buffer.get_velocity_stats(p1["id"])

            for p2 in people[i+1:]:
                p2_center = np.array([(p2["box"][0]+p2["box"][2])/2, (p2["box"][1]+p2["box"][3])/2])
                cx2, cy2 = int(np.clip(p2_center[0], 0, w-1)), int(np.clip(p2_center[1], 0, h-1))
                p2_depth = depth_map[cy2, cx2]

                dist2d = np.linalg.norm(p1_center - p2_center)
                
                # Check Proximity + Depth plane
                if dist2d < self.dist_thresh and abs(p1_depth - p2_depth) < self.depth_thresh:
                    # Check Motion map (Optical Flow) for "Chaos" in the interaction zone
                    mid = ((p1_center + p2_center) / 2).astype(int)
                    y1, y2 = max(0, mid[1]-40), min(h, mid[1]+40)
                    x1, x2 = max(0, mid[0]-40), min(w, mid[0]+40)
                    roi = motion_map[y1:y2, x1:x2]
                    
                    if roi.size > 0 and np.mean(roi) > self.motion_thresh:
                        alerts.append("ALERT: PHYSICAL ALTERCATION")
        return alerts

class LoiteringDetector:
    def __init__(self, limit_sec=20):
        self.limit = limit_sec
        self.entry_times = {}

    def check(self, data, *args):
        now = time.time()
        alerts = []
        current_ids = [p["id"] for p in data["people"]]
        for p_id in current_ids:
            if p_id not in self.entry_times: self.entry_times[p_id] = now
            elif now - self.entry_times[p_id] > self.limit:
                alerts.append(f"SUSPICIOUS: LOITERING (ID:{p_id})")
        self.entry_times = {pid: t for pid, t in self.entry_times.items() if pid in current_ids}
        return alerts

class IntrusionDetector:
    def __init__(self, tripwire_y_ratio=0.7):
        self.ratio = tripwire_y_ratio
        self.last_y = {}

    def check(self, data, motion_map, *args):
        alerts = []
        h = motion_map.shape[0]
        line_y = int(h * self.ratio)
        current_ids = [p["id"] for p in data["people"]]
        for p in data["people"]:
            p_id, curr_y = p["id"], p["box"][3]
            if p_id in self.last_y:
                if self.last_y[p_id] < line_y and curr_y >= line_y:
                    alerts.append(f"ALERT: INTRUSION (ID:{p_id})")
            self.last_y[p_id] = curr_y
        self.last_y = {pid: y for pid, y in self.last_y.items() if pid in current_ids}
        return alerts

class AbandonedObjectDetector:
    def __init__(self, limit_sec=15, dist_thresh=120, depth_thresh=0.05):
        self.limit = limit_sec
        self.dist = dist_thresh
        self.depth_thresh = depth_thresh
        self.item_timers = {}

    def check(self, data, motion_map, depth_map):
        if not data.get("items"): return []
        if depth_map.max() > 1.0:
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-5)
            
        now = time.time()
        alerts = []
        for item in data["items"]:
            item_center = np.array([(item["box"][0]+item["box"][2])/2, (item["box"][1]+item["box"][3])/2])
            ix, iy = int(item_center[0]), int(item_center[1]) # Scaled to AI resolution
            
            is_attended = False
            for p in data["people"]:
                p_center = np.array([(p["box"][0]+p["box"][2])/2, (p["box"][1]+p["box"][3])/2])
                dist2d = np.linalg.norm(item_center - p_center)
                if dist2d < self.dist:
                    # Verify they are on the same depth plane
                    px, py = int(p_center[0]), int(p_center[1])
                    if abs(depth_map[iy, ix] - depth_map[py, px]) < self.depth_thresh:
                        is_attended = True
                        break
            
            if not is_attended:
                if item["id"] not in self.item_timers: self.item_timers[item["id"]] = now
                elif now - self.item_timers[item["id"]] > self.limit:
                    alerts.append(f"WARNING: UNATTENDED OBJECT ({item['id']})")
            else: self.item_timers.pop(item["id"], None)
        return alerts

class FallDetector:
    def __init__(self, time_to_confirm=1.5):
        self.prone_timers = {}
        self.confirm_time = time_to_confirm

    def check(self, results, *args):
        alerts = []
        if not results or not results.get('pose'):
            return alerts

        pose_data = results['pose']
        # Extract boxes and keypoints from YOLO Pose result
        boxes = pose_data.boxes
        keypoints = pose_data.keypoints.xy.cpu().numpy() # [N, 17, 2]

        if boxes is None or len(boxes) == 0:
            return alerts

        current_ids = []
        for i, box in enumerate(boxes):
            # 1. Get ID (fallback to index if no tracker)
            pid = int(box.id[0]) if box.id is not None else i
            current_ids.append(pid)

            # 2. Geometry Check (Aspect Ratio)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w, h = x2 - x1, y2 - y1
            is_horizontal = w > h * 1.2

            # 3. Pose Check (Head vs Ankle Height)
            # YOLO Keypoints: 0=Nose, 15=L-Ankle, 16=R-Ankle
            kpts = keypoints[i]
            nose_y = kpts[0][1]
            ankle_y = max(kpts[15][1], kpts[16][1])
            
            # If the head is very close to the ankle level vertically
            is_low_altitude = False
            if nose_y > 0 and ankle_y > 0:
                # If nose is in the bottom 30% of the body box
                is_low_altitude = nose_y > (y1 + h * 0.7)

            # 4. Final Logic: horizontal + low head = confirmed prone position
            if is_horizontal or is_low_altitude:
                if pid not in self.prone_timers:
                    self.prone_timers[pid] = time.time()
                elif time.time() - self.prone_timers[pid] > self.confirm_time:
                    alerts.append(f"CRITICAL: FALL DETECTED (ID:{pid})")
            else:
                self.prone_timers.pop(pid, None)

        # Cleanup old IDs
        self.prone_timers = {pid: t for pid, t in self.prone_timers.items() if pid in current_ids}
        return alerts