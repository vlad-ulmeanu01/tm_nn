import pandas as pd
import numpy as np
import copy

dfr = pd.read_csv("/home/vlad/Desktop/Probleme/Trackmania/MakeUnrefinedKb/merged_unrefined_kb.csv", skipinitialspace = True)

fout = open("/home/vlad/Desktop/Probleme/Trackmania/export_pts_kb_aug.txt", "w")

totalLen = 0
def writeArr(fout, name: str, arr: list):
    global totalLen
    totalLen += len(arr)
    fout.write(f"{name} {round(min(arr), 3)} {round(max(arr), 3)}\n")

for ch in ['x', 'y', 'z']:
    if ch == 'x':
        for i in range(21): #ma ocup si de augumentare. flip la coeficientii lui x tb sa rezulte in flip la steer.
            arr = copy.deepcopy(list(dfr[f"coef_{ch}{i}"]))
            arr.extend(list(np.array(dfr[f"coef_{ch}{i}"]) * -1))
            writeArr(fout, f"{ch}{i}", arr)
    else:
        for i in range(21):
            writeArr(fout, f"{ch}{i}", dfr[f"coef_{ch}{i}"])

writeArr(fout, "timeSinceLastBrake", dfr["timeSinceLastBrake"])
writeArr(fout, "timeSpentBraking", dfr["timeSpentBraking"])
writeArr(fout, "timeSinceLastAir", dfr["timeSinceLastAir"])
writeArr(fout, "timeSpentAir", dfr["timeSpentAir"])

fout.close()

print(f"totalLen = {totalLen}.")
