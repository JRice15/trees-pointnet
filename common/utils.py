import time

class MyTimer:

    def __init__(self, msg_indent=2, decimals=4):
        self.msg_indent = msg_indent
        self.decimals = decimals
        self.start()
    
    def start(self):
        self._start_time = time.perf_counter()
    
    def measure(self, name=None):
        elapsed = time.perf_counter() - self._start_time
        prefix = " " * self.msg_indent
        if name is not None:
            prefix += str(name) + ": "
        if elapsed > 60:
            elapsed /= 60
            unit = "min"
        else:
            unit = "sec"
        elapsed = round(elapsed, self.decimals)
        print(f"{prefix}{elapsed} {unit}")
        self.start()
