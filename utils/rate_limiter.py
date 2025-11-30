import threading
import time
from collections import deque

# Global rate limiter (500 RPM)
class RateLimiter:
    """
    Thread-safe rate limiter that ensures no more than `max_calls_per_minute` calls
    """

    def __init__(self, max_calls_per_minute: int):
        self.max_calls_per_minute = max_calls_per_minute
        self.lock = threading.Lock()
        self.call_timestamps = deque() 

    def wait_for_slot(self):
        """
        Block until making another call will not exceed `max_calls_per_minute`
        in the last 60 seconds.
        """
        while True:
            with self.lock:
                now = time.time()

                while self.call_timestamps and now - self.call_timestamps[0] > 60:
                    self.call_timestamps.popleft()

                if len(self.call_timestamps) < self.max_calls_per_minute:
                    self.call_timestamps.append(now)
                    return

                earliest = self.call_timestamps[0]
                wait_time = 60 - (now - earliest)

            if wait_time > 0:
                time.sleep(wait_time)
            else:
                time.sleep(0.01)