import numpy as np

neg_exp_table = [np.exp(x) for x in np.linspace(-100, 0, 10**6)]
def getNegExp(x):
    return 0.0 if x < -100 else neg_exp_table[max(0, min(len(neg_exp_table) - 1, int(x * 1e4 + 1000000)))]

shifts = [100, 162, 235, 342, 501]

"""
vreau viteza x in km/h.
"""
def refineSpeed(x: float) -> list:
    x = max(min(500.0, x), 0.0)

    #0 -- 100 -- 162 -- 235 -- 342 -- 500.
    sol = [0.0] * 5
    i = 0
    while i < 5 and x > shifts[i]:
        sol[i] = 1
        i += 1

    sol[i] = x / shifts[i] if i == 0 else (x - shifts[i-1]) / (shifts[i] - shifts[i-1])
    return sol

def reverseGetSimpleSteer(output):
    ind = np.argmax(output)
    return -65536 if ind == 0 else (0 if ind == 1 else 65536)

class Refiner:
    class Entry:
        def __init__(self, tip: str, params: tuple):
            self.tip = tip
            self.params = params

    def __init__(self, fname: str):
        self.ht = {}

        fin = open(fname)
        for line in fin.readlines():
            line = line.strip()
            if len(line) and line[0] != '#':
                arr = line.split()
                if arr[0] == "lg":
                    self.ht[arr[1]] = self.Entry("lg", (int(arr[2]), int(arr[3])))
                elif arr[0] == "raport":
                    self.ht[arr[1]] = self.Entry("raport", (int(arr[2]),))
                else:
                    assert(False)

        fin.close()

    def refineValue(self, nume: str, val: float) -> list:
        assert(nume in self.ht)
        if self.ht[nume].tip == "lg":
            a, b = self.ht[nume].params
            sol = [0.0] * 2
            ind = 0 if val >= 0 else 1
            sol[ind] = min(1.0, (a + np.log10(max(10 ** (-a), abs(val)))) / b)
            return sol
        elif self.ht[nume].tip == "raport":
            a, = self.ht[nume].params
            sol = [0.0] * 2
            ind = 0 if val >= 0 else 1
            sol[ind] = min(1.0, abs(val) / a)
            return sol
        else:
            assert(False)