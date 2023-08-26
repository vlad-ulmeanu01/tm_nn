import pandas as pd
import numpy as np
import copy
import os
import refine_utils

dfr = pd.read_csv("/home/vlad/Desktop/Probleme/Trackmania/merged_unrefined.csv", skipinitialspace = True)

coefs, augCoefs = [], []
szCoefs = [0] * 3
for ch in ['x', 'y', 'z']:
    if ch == 'x':
        for i in range(21): #ma ocup si de augumentare. flip la coeficientii lui x tb sa rezulte in flip la steer.
            arr = copy.deepcopy(list(dfr[f"coef_{ch}{i}"]))
            arr.extend(list(np.array(dfr[f"coef_{ch}{i}"]) * -1))

            coefs.append(refine_utils.refineValues(arr))
            szCoefs[ord(ch) - ord('x')] += len(coefs[-1][0])
            print(f"{ch}{i} {len(coefs[-1][0])}")
    else:
        for i in range(21):
            coefs.append(refine_utils.refineValues(dfr[f"coef_{ch}{i}"]))
            szCoefs[ord(ch) - ord('x')] += len(coefs[-1][0])
            print(f"{ch}{i} {len(coefs[-1][0])}")

print(szCoefs)

fout = open("/home/vlad/Desktop/Probleme/Trackmania/refined_fix.csv", "w")

#viteza.
fout.write(''.join([f"v{i}, " for i in range(5)]))

#materialul.
fout.write(''.join([f"m{i}, " for i in range(4)]))

#brake/air timeSince/timeSpent.
refTimeSinceLastBrake = refine_utils.refineValues(dfr["timeSinceLastBrake"]) #3 puncte.
refTimeSpentBraking = refine_utils.refineValues(dfr["timeSpentBraking"]) #2
refTimeSinceLastAir = refine_utils.refineValues(dfr["timeSinceLastAir"]) #3
refTimeSpentAir = refine_utils.refineValues(dfr["timeSpentAir"]) #2

fout.write("b00, b01, b02, b10, b11, a00, a01, a02, a10, a11, ")

#coeficientii pentru x, y, z.
fout.write(''.join([f"c{i}, " for i in range(sum(szCoefs))]))

#gas, brake.
fout.write("gas, brake, ")

#steer.
refSteer = refine_utils.refineValues(
    dfr["steer"],
    m = refine_utils.MIN_STEER,
    M = refine_utils.MAX_STEER,
    cntIntervals = refine_utils.CNT_INTERVALS_STEER,
    maxDistToPoint = refine_utils.STEER_MAX_DIST_TO_POINT
)

refAugSteer = refine_utils.refineValues(
    list(np.array(dfr["steer"]) * -1),
    m = refine_utils.MIN_STEER,
    M = refine_utils.MAX_STEER,
    cntIntervals = refine_utils.CNT_INTERVALS_STEER,
    maxDistToPoint = refine_utils.STEER_MAX_DIST_TO_POINT
)

fout.write(''.join([f"s{i}, " for i in range(refine_utils.CNT_INTERVALS_STEER + 1)])[:-2] + "\n")

n = len(dfr["vx"])

for i in range(n):
    for aug in range(2): #aug == 0/1 <=> normal, flip x/steer.
        #viteza.
        arrSpeed = refine_utils.refineSpeed(np.linalg.norm(np.array([dfr["vx"][i], dfr["vy"][i], dfr["vz"][i]])) * 3.6)
        fout.write(''.join([f"{round(a, 3)}, " for a in arrSpeed]))

        #materiale.
        fout.write(f"{dfr['road'][i]}, {dfr['air'][i]}, {dfr['dirt'][i]}, {dfr['grass'][i]}, ")

        #brake timeSince.
        fout.write(''.join([f"{round(refTimeSinceLastBrake[i][j], 3)}, " for j in range(len(refTimeSinceLastBrake[i]))]))

        #brake timeSpent.
        fout.write(''.join([f"{round(refTimeSpentBraking[i][j], 3)}, " for j in range(len(refTimeSpentBraking[i]))]))

        #air timeSince.
        fout.write(''.join([f"{round(refTimeSinceLastAir[i][j], 3)}, " for j in range(len(refTimeSinceLastAir[i]))]))

        #air timeSpent.
        fout.write(''.join([f"{round(refTimeSpentAir[i][j], 3)}, " for j in range(len(refTimeSpentAir[i]))]))

        #coeficienti (x/y/z).
        for j in range(21 * 3):
            if j < 21: #afisez flip x in loc de normal.
                if aug == 0:
                    fout.write(''.join([f"{round(coefs[j][i][z], 3)}, " for z in range(len(coefs[j][i]))]))
                else:
                    fout.write(''.join([f"{round(coefs[j][i + n][z], 3)}, " for z in range(len(coefs[j][i + n]))]))
            else:
                fout.write(''.join([f"{round(coefs[j][i][z], 3)}, " for z in range(len(coefs[j][i]))]))

        #output: gas, brake.
        fout.write(f"{dfr['gas'][i]}, {dfr['brake'][i]}, ")

        #output: steer.
        if aug == 0:
            fout.write(''.join([f"{round(refSteer[i][j], 3)}, " for j in range(len(refSteer[i]))])[:-2] + "\n")
        else:
            fout.write(''.join([f"{round(refAugSteer[i][j], 3)}, " for j in range(len(refAugSteer[i]))])[:-2] + "\n")

    print(f"{i+1}/{n} ok.")

fout.close()