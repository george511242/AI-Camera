import time


class FPSCalculator:
    start_time: float
    
    def set_start_time(self, time: float):
        self.start_time = time
        
    def calculate(self):
        inference_time = max(time.time() - self.start_time, 0.0001)
        fps = round(1 / inference_time, 1) 
        return fps