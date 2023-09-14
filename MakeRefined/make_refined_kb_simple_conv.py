import pandas as pd
import numpy as np
import copy
import os
import refine_utils

dfr = pd.read_csv(???, skipinitialspace = True)

cnt_coef_axis = 97

norm = {"x": 131.108, "y": 170.909, "z": 127.232}
norm_timeSinceLastBrake = 69610
norm_timeSpentBraking = 2380
norm_timeSinceLastAir = 18330
norm_timeSpentAir = 5210

fout = open("??/refined_kb_simple_conv_noaug.csv", "w")

#viteza.
fout.write(''.join([f"v{i}, " for i in range(5)]))

#materialul.
fout.write(''.join([f"m{i}, " for i in range(4)]))

#brake/air timeSince/timeSpent.
fout.write("b0, b1, a0, a1, ")

#coeficientii pentru x, y, z.
fout.write(''.join([f"c{i}, " for i in range(cnt_coef_axis * 3 * 2)]))

#gas, brake.
fout.write("gas, brake, ")

#steer.
fout.write("s_left, s_straight, s_right\n")

refSteer = [[1, 0, 0] if s == -65536 else ([0, 1, 0] if s == 0 else [0, 0, 1]) for s in dfr["steer"]]

n = len(dfr["vx"])

for i in range(n):
    #viteza.
    arrSpeed = refine_utils.refineSpeed(np.linalg.norm(np.array([dfr["vx"][i], dfr["vy"][i], dfr["vz"][i]])) * 3.6)
    fout.write(''.join([f"{round(a, 3)}, " for a in arrSpeed]))

    #materiale.
    fout.write(f"{dfr['road'][i]}, {dfr['air'][i]}, {dfr['dirt'][i]}, {dfr['grass'][i]}, ")

    fout.write(f"{round(dfr['timeSinceLastBrake']][i] / norm_timeSinceLastBrake, 3)}, ") #brake timeSince.
    fout.write(f"{round(dfr['timeSpentBraking']][i] / norm_timeSpentBraking, 3)}, ") #brake timeSpent.
    fout.write(f"{round(dfr['timeSinceLastAir']][i] / norm_timeSinceLastAir, 3)}, ") #air timeSince.
    fout.write(f"{round(dfr['timeSpentAir']][i] / norm_timeSpentAir, 3)}, ") #air timeSpent.

    #coeficienti (x/y/z).
    for ch in ['x', 'y', 'z']
        for _ in range(cnt_coef_axis):
            val = dfr[f'coef_{ch}{j}'][i] / norm[ch]
            arr = [val, 0.0] if val >= 0 else [0.0, -val]
            fout.write(f"{round(arr[0], 3)}, {round(arr[1], 3)}, ")

    #output: gas, brake.
    fout.write(f"{dfr['gas'][i]}, {dfr['brake'][i]}, ")

    #output: steer.
    fout.write(''.join([f"{refSteer[i][j]}, " for j in range(len(refSteer[i]))])[:-2] + "\n")

    print(f"{i+1}/{n} ok.")

fout.close()
