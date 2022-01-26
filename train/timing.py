import time

class Timing:

    def __init__(self, stats, name):
        self.stats = stats
        self.name = name
        if name not in self.stats:
            self.stats[self.name] = dict(time=0.0, count=0)

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        self.stats[self.name]['time'] += (time.time() - self.start_time)
        self.stats[self.name]['count'] += 1


class MultiTiming:
    def __init__(self, stats):
        self.stats = stats

    def start(self, name):
        if name not in self.stats:
            self.stats[name] = dict(time=0.0, count=0, start_time=time.time())

        self.stats[name]['start_time'] = time.time()

    def end(self, name):
        duration = time.time() - self.stats[name]['start_time']
        self.stats[name]['time'] += duration
        self.stats[name]['count'] += 1

        return duration


def pprint_stats(stats):
    durations = { k: v['time'] / v['count'] for k, v in stats.items() }

    for name, duration in durations.items():
        print(f'{name}: {duration:.3f}', flush=True)