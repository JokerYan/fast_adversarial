import time


class Timer:
    def __init__(self):
        self.total_time = 0
        self.last_start = None  # last starting time

    def start(self):
        assert self.last_start is None
        self.last_start = time.time()

    def end(self):
        assert self.last_start is not None
        elapsed = time.time() - self.last_start
        self.total_time += elapsed


# key: Timer()
timer_dict = {}


def start_timer(key):
    if key not in timer_dict:
        timer_dict[key] = Timer()
    timer_dict[key].start()


def end_timer(key):
    assert key in timer_dict
    timer_dict[key].end()


def get_total_time(key):
    assert key in timer_dict
    return timer_dict[key].total_time
