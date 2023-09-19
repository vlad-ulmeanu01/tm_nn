import numpy as np

neg_exp_table = [np.exp(x) for x in np.linspace(-100, 0, 10**6)]
def getNegExp(x):
    return 0.0 if x < -100 else neg_exp_table[max(0, min(len(neg_exp_table) - 1, int(x * 1e4 + 1000000)))]

"""
vreau viteza x in km/h.
"""
def refineSpeed(x: float) -> list:
    shifts = [100, 162, 235, 342, 501]
    x = max(min(500.0, x), 0.0)

    #0 -- 100 -- 162 -- 235 -- 342 -- 500.
    sol = [0.0] * 5
    i = 0
    while i < 5 and x > shifts[i]:
        sol[i] = 1
        i += 1

    sol[i] = x / shifts[i] if i == 0 else (x - shifts[i-1]) / (shifts[i] - shifts[i-1])
    return sol

class Refiner:
    class Entry:
        def __init__(self, avg: float, std: float, fixedPoints: list):
            self.avg = avg
            self.std = std
            self.fixedPoints = fixedPoints

    def __init__(self, fname: str):
        self.ht = {}

        fin = open(fname)
        for line in fin.readlines():
            line = line.strip()
            if len(line) and line[0] != '#':
                arr = line.split()
                self.ht[arr[0]] = self.Entry(float(arr[1]), float(arr[2]), list(map(float, arr[3:])))
        fin.close()

    def refineValue(self, tip: str, val: float) -> list:
        assert(tip in self.ht)
        val = (val - self.ht[tip].avg) / self.ht[tip].std #normalizare.
        val = max(self.ht[tip].fixedPoints[0], min(self.ht[tip].fixedPoints[-1], val)) #clipping.
        return [getNegExp(-(val - p) ** 2) for p in self.ht[tip].fixedPoints]