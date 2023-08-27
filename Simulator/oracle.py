import pandas as pd
import copy

"""
oracolul primeste input din simulare si decide care este urmatoarea actiune.
"""
class Oracle:
    def __init__(self):
        self.dir = 1000
        self.last_steer = 0

        self.state_series = []

        #TODO racing line pentru oracolul normal.

    """
    primesc informatie noua de la worker.
    """
    def update_state_series(self, state_series: list):
        print(f"(oracle) new length: {len(state_series)}.")
        self.state_series = copy.deepcopy(state_series)

    """
    tinand cont de state_series, prezice urmatorul input, tuplu (steer, gas, brake).
    """
    def predict(self):
        if abs(self.last_steer + self.dir) > 65536:
            self.dir *= -1

        self.last_steer += self.dir
        return (self.last_steer, 1, 0)