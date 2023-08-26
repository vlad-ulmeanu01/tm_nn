import pandas as pd
import numpy as np
import copy

import refine_utils

dfr = pd.read_csv("/home/vlad/Desktop/Probleme/Trackmania/merged_unrefined.csv", skipinitialspace = True)
fout = open("export_pts.txt", "w")

def writeArr(fout, name: str, arr: list):
    fout.write(f"{name} {len(arr)} " + ''.join([f"{round(a, 3)} " for a in arr]) + "\n")

for ch in ['x', 'y', 'z']:
    if ch == 'x':
        for i in range(21): #ma ocup si de augumentare. flip la coeficientii lui x tb sa rezulte in flip la steer.
            arr = copy.deepcopy(list(dfr[f"coef_{ch}{i}"]))
            arr.extend(list(np.array(dfr[f"coef_{ch}{i}"]) * -1))
            writeArr(fout, f"{ch}{i}", refine_utils.refineGetPoints(arr)[0])
    else:
        for i in range(21):
            writeArr(fout, f"{ch}{i}", refine_utils.refineGetPoints(dfr[f"coef_{ch}{i}"])[0])

writeArr(fout, "timeSinceLastBrake", refine_utils.refineGetPoints(dfr["timeSinceLastBrake"])[0])
writeArr(fout, "timeSpentBraking", refine_utils.refineGetPoints(dfr["timeSpentBraking"])[0])
writeArr(fout, "timeSinceLastAir", refine_utils.refineGetPoints(dfr["timeSinceLastAir"])[0])
writeArr(fout, "timeSpentAir", refine_utils.refineGetPoints(dfr["timeSpentAir"])[0])

fout.close()