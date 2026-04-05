import time
from collections import defaultdict

class BehaviorAccumulator:
    def __init__(self, window_sec=10, count_threshold=3):
        self.window_sec = window_sec
        self.threshold = count_threshold
        # Stores { event_text: [timestamp1, timestamp2...] }
        self.event_history = defaultdict(list)
        # Prevents saving multiple videos for the same persistent event
        self.confirmed_events = set()
        self.last_clean_time = time.time()

    def add_and_check(self, alert_text):
        now = time.time()
        
        # Clean history periodically to save memory
        if now - self.last_clean_time > self.window_sec:
            self._clean_old_events(now)
            self.last_clean_time = now

        # Add current event
        self.event_history[alert_text].append(now)

        # Count events in current window
        recent_events = [t for t in self.event_history[alert_text] if now - t < self.window_sec]
        self.event_history[alert_text] = recent_events # Update with cleaned list

        if len(recent_events) >= self.threshold:
            if alert_text not in self.confirmed_events:
                self.confirmed_events.add(alert_text)
                return True # Trigger Video Save
        
        return False

    def _clean_old_events(self, now):
        for key in list(self.event_history.keys()):
            self.event_history[key] = [t for t in self.event_history[key] if now - t < self.window_sec]
            if not self.event_history[key]:
                del self.event_history[key]
                self.confirmed_events.discard(key) # Reset confirmation if event stopped