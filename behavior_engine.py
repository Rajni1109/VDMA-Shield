from behavior_modules import ViolenceDetector, LoiteringDetector, IntrusionDetector, AbandonedObjectDetector, FallDetector

class BehaviorEngine:
    def __init__(self):
        # Configuration - Parameters tuned for high-sensitivity detection
        self.modules = [                
            ViolenceDetector(motion_thresh=45, dist_thresh=280, depth_thresh=100),
            LoiteringDetector(limit_sec=300),
            IntrusionDetector(tripwire_y_ratio=420),
            AbandonedObjectDetector(limit_sec=15, dist_thresh=120, depth_thresh=30),
            FallDetector() 
        ]

    def analyze(self, results_dict, motion_map, depth_map):
        # 1. Unpack the dictionary
        pose_results = results_dict.get('pose')
        det_results = results_dict.get('detection')

        # 2. Check for tracked people (from pose model)
        if pose_results is None or not pose_results.boxes or pose_results.boxes.id is None:
            return []
        
        # 3. Update data parsing logic to handle separate pose and detection results
        data = self._parse_results_dict(pose_results, det_results)
        
        all_alerts = []
        for module in self.modules:
            # Pass data and maps to each module
            alerts = module.check(data, motion_map, depth_map)
            if alerts: 
                all_alerts.extend(alerts)

        # Remove duplicate alerts
        return list(set(all_alerts))

    def _parse_results_dict(self, pose_results, det_results):
        """Parses combined pose and detection results into structured data."""                
        
        # --- Process Pose Results (People) ---
        people = []
        if pose_results and pose_results.boxes.id is not None:
            p_boxes = pose_results.boxes.xyxy.cpu().numpy()
            p_ids = pose_results.boxes.id.cpu().numpy().astype(int)
            p_kpts = pose_results.keypoints.xy.cpu().numpy() if pose_results.keypoints is not None else None
            
            for i in range(len(p_ids)):
                people.append({
                    "id": p_ids[i],
                    "box": p_boxes[i],
                    "kpts": p_kpts[i] if p_kpts is not None else None
                })

        # --- Process Detection Results (Items) ---
        items = []
        if det_results and det_results.boxes.id is not None:
            d_boxes = det_results.boxes.xyxy.cpu().numpy()
            d_ids = det_results.boxes.id.cpu().numpy().astype(int)
            d_cls = det_results.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(d_ids)):
                items.append({
                    "id": d_ids[i],
                    "box": d_boxes[i],
                    "class": d_cls[i] # Useful for behavior modules
                })
        
        return {"people": people, "items": items}