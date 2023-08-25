import os

fout = open("merged_unrefined.csv", "w")

fout.write("vx, vy, vz, road, air, dirt, grass, timeSinceLastBrake, timeSpentBraking, timeSinceLastAir, timeSpentAir, coef_x0, coef_x1, coef_x2, coef_x3, coef_x4, coef_x5, coef_x6, coef_x7, coef_x8, coef_x9, coef_x10, coef_x11, coef_x12, coef_x13, coef_x14, coef_x15, coef_x16, coef_x17, coef_x18, coef_x19, coef_x20, coef_y0, coef_y1, coef_y2, coef_y3, coef_y4, coef_y5, coef_y6, coef_y7, coef_y8, coef_y9, coef_y10, coef_y11, coef_y12, coef_y13, coef_y14, coef_y15, coef_y16, coef_y17, coef_y18, coef_y19, coef_y20, coef_z0, coef_z1, coef_z2, coef_z3, coef_z4, coef_z5, coef_z6, coef_z7, coef_z8, coef_z9, coef_z10, coef_z11, coef_z12, coef_z13, coef_z14, coef_z15, coef_z16, coef_z17, coef_z18, coef_z19, coef_z20, gas, brake, steer\n")

for path, subdirs, files in os.walk("./csvs/"):
    for file in files:
        fnow = open("./csvs/" + file, "r")
        for line in fnow.readlines()[1:]:
            if line.count(',') == 76:
                fout.write(line)
        fnow.close()

fout.close()