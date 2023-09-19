import pandas as pd
import numpy as np
import copy
import os
import refine_utils

dfr = pd.read_csv("/home/vlad/Desktop/Probleme/Trackmania/MakeUnrefinedKb/merged_unrefined_kb.csv", skipinitialspace = True)

coefs, augCoefs = [], []
szCoefs = [0] * 3
for ch in ['x', 'y', 'z']:
    if False: #ch == 'x':
        for i in range(21): #ma ocup si de augumentare. flip la coeficientii lui x tb sa rezulte in flip la steer.
            arr = copy.deepcopy(list(dfr[f"coef_{ch}{i}"]))
            arr.extend(list(np.array(dfr[f"coef_{ch}{i}"]) * -1))

            coefs.append(refine_utils.refineValuesSimpleKb(arr))
            szCoefs[ord(ch) - ord('x')] += len(coefs[-1][0])
            print(f"{ch}{i} {len(coefs[-1][0])}")
    else:
        for i in range(21):
            coefs.append(refine_utils.refineValuesSimpleKb(dfr[f"coef_{ch}{i}"]))
            szCoefs[ord(ch) - ord('x')] += len(coefs[-1][0])
            print(f"{ch}{i} {len(coefs[-1][0])}")

print(szCoefs)

fout = open("/home/vlad/Desktop/Probleme/Trackmania/refined_kb_simple_noaug.csv", "w")

#viteza.
fout.write(''.join([f"v{i}, " for i in range(5)]))

#materialul.
fout.write(''.join([f"m{i}, " for i in range(4)]))

#brake/air timeSince/timeSpent.
M = max(dfr["timeSinceLastBrake"]); refTimeSinceLastBrake = [x / M for x in dfr["timeSinceLastBrake"]]
M = max(dfr["timeSpentBraking"]); refTimeSpentBraking = [x / M for x in dfr["timeSpentBraking"]]
M = max(dfr["timeSinceLastAir"]); refTimeSinceLastAir = [x / M for x in dfr["timeSinceLastAir"]]
M = max(dfr["timeSpentAir"]); refTimeSpentAir = [x / M for x in dfr["timeSpentAir"]]

fout.write("b0, b1, a0, a1, ")

#coeficientii pentru x, y, z.
fout.write(''.join([f"c{i}, " for i in range(sum(szCoefs))]))

#gas, brake.
fout.write("gas, brake, ")

#steer.
refSteer = [[1, 0, 0] if s == -65536 else ([0, 1, 0] if s == 0 else [0, 0, 1]) for s in dfr["steer"]]
refAugSteer = [[1, 0, 0] if -s == -65536 else ([0, 1, 0] if s == 0 else [0, 0, 1]) for s in dfr["steer"]]

fout.write("s_left, s_straight, s_right\n")

n = len(dfr["vx"])

for i in range(n):
    for aug in [0]: #range(2): #aug == 0/1 <=> normal, flip x/steer. #[0]:
        #viteza.
        arrSpeed = refine_utils.refineSpeed(np.linalg.norm(np.array([dfr["vx"][i], dfr["vy"][i], dfr["vz"][i]])) * 3.6)
        fout.write(''.join([f"{round(a, 3)}, " for a in arrSpeed]))

        #materiale.
        fout.write(f"{dfr['road'][i]}, {dfr['air'][i]}, {dfr['dirt'][i]}, {dfr['grass'][i]}, ")

        fout.write(f"{round(refTimeSinceLastBrake[i], 3)}, ") #brake timeSince.
        fout.write(f"{round(refTimeSpentBraking[i], 3)}, ") #brake timeSpent.
        fout.write(f"{round(refTimeSinceLastAir[i], 3)}, ") #air timeSince.
        fout.write(f"{round(refTimeSpentAir[i], 3)}, ") #air timeSpent.

        #coeficienti (x/y/z).
        for j in range(21 * 3):
            if j < 21:
                if aug == 0:
                    fout.write(''.join([f"{round(coefs[j][i][z], 3)}, " for z in range(len(coefs[j][i]))]))
                else: #afisez flip x in loc de normal.
                    fout.write(''.join([f"{round(coefs[j][i + n][z], 3)}, " for z in range(len(coefs[j][i + n]))]))
            else:
                fout.write(''.join([f"{round(coefs[j][i][z], 3)}, " for z in range(len(coefs[j][i]))]))

        #output: gas, brake.
        fout.write(f"{dfr['gas'][i]}, {dfr['brake'][i]}, ")

        #output: steer.
        if aug == 0:
            fout.write(''.join([f"{refSteer[i][j]}, " for j in range(len(refSteer[i]))])[:-2] + "\n")
        else:
            fout.write(''.join([f"{refAugSteer[i][j]}, " for j in range(len(refAugSteer[i]))])[:-2] + "\n")

    print(f"{i+1}/{n} ok.")

fout.close()
