import sklearn.linear_model
import sklearn.neighbors
import pandas as pd
import numpy as np
import time as ty
import copy

import nr_utils

"""
primesc doua replay-uri, R1 si R2.
urmaresc punctele din R1. R2 este folosit ca racing line.
vreau sa construiesc un .csv care sa fie folosit ca input pentru antrenarea retelei.
"""

"""
vreau sa mut originea sistemului in punctul (x, y, z), cu orientarea catre (yaw, pitch, roll).
am n puncte in plan, organizate in vectorii xs, ys, zs.
"""

def moveOriginTo(newOrigin: tuple, n: int, xs: list, ys: list, zs: list) -> tuple:
    xs, ys, zs = copy.deepcopy(xs), copy.deepcopy(ys), copy.deepcopy(zs)

    #in general, XOY e planul pe pamant si Z iese din el. aici am XOZ plan pe pamant cu Y care iese din el.
    #astfel, am pitch-ul si yaw-ul interschimbate la initializare mai jos, dar ele isi pastreaza intelesul:
    #yaw: viraj,
    #pitch: urcare,
    #roll: rotisor.

    x, y, z, yaw, pitch, roll = newOrigin

    ca, sa = np.cos(-pitch), np.sin(-pitch)
    matYaw = np.array([[ca, -sa, 0, 0], [sa, ca, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    cb, sb = np.cos(-yaw), np.sin(-yaw)
    matPitch = np.array([[cb, 0, sb, 0], [0, 1, 0, 0], [-sb, 0, cb, 0], [0, 0, 0, 1]])

    cc, sc = np.cos(-roll), np.sin(-roll)
    matRoll = np.array([[1, 0, 0, 0], [0, cc, -sc, 0], [0, sc, cc, 0], [0, 0, 0, 1]])

    matTrans = np.array([[1, 0, 0, -x], [0, 1, 0, -y], [0, 0, 1, -z], [0, 0, 0, 1]])

    mat = matYaw @ matPitch @ matRoll @ matTrans

    matPoints = np.ones((4, n))
    for i in range(n):
        matPoints[0][i], matPoints[1][i], matPoints[2][i] = xs[i], ys[i], zs[i]

    matPoints = mat @ matPoints
    for i in range(n):
        xs[i], ys[i], zs[i] = matPoints[0][i], matPoints[1][i], matPoints[2][i]

    return xs, ys, zs

"""
v[i] = v[i] * (1 - alfa) + v[i-1] * alfa pt orice i >= l.
"""
def exponentialSmooth(v: list, l: int) -> list:
    v = copy.deepcopy(v)
    alfa, mulBy = 0.9, 0.98
    for i in range(max(l, 1), len(v)):
        v[i] = v[i] * (1 - alfa) + v[i-1] * alfa
        alfa *= mulBy

    return v

"""
intoarce v[i], l <= i < r, abs(v[i]) >= abs(v[j]) \forall l <= j < r.
"""
def minMaxPool(v, l: int, r: int) -> float:
    best = l
    for i in range(l+1, r):
        if abs(v[best]) < abs(v[i]):
            best = i
    return v[best]

"""
aplica smoothing, crestere momentana, convolutie 1D.
"""
def timeSeriesModify(v: list) -> list:
    assert(len(v) == 300)

    #dau smooth de la jumatate incolo. ptc am 1.5s din trecut si 1.5s din racing line, dau smooth doar peste viitor.
    v = exponentialSmooth(v, int(len(v) // 2))

    #transform vectorul in diferente.
    u = [0.0]
    for i in range(1, len(v)):
        u.append(v[i] - v[i-1])

    v = np.convolve(u, [1] * 10, "valid") #acum len(v) = 300 - 10 + 1.
    return [minMaxPool(v, i, i+3) for i in range(0, len(v), 3)]

"""
primeste 2 nume de fisiere csv. intoarce numele csv-ului in care au fost scrise niste inputuri nerafinate pentru retea.
mulCoef: in acest format coeficientii sunt destul de mici. round(.., 3) ii cam baga in 0. coef bagati in .csv ca round(.. * mulCoef, 3).
"""
def make_input_from_pair(fName1: str, fName2: str, mulCoef = 1000) -> str:
    dfr = [pd.read_csv(fName1, skipinitialspace = True), pd.read_csv(fName2, skipinitialspace = True)]

    outName = str(int(ty.time() * 10000)) + ".csv"
    fout = open(outName, "w")
    fout.write(f"vx, vy, vz, road, air, dirt, grass, timeSinceLastBrake, timeSpentBraking, timeSinceLastAir, timeSpentAir, " +
               ''.join(f"coef_x{i}, " for i in range(97)) + ''.join(f"coef_y{i}, " for i in range(97)) + ''.join(f"coef_z{i}, " for i in range(97)) +
               "gas, brake, steer" + "\n")

    #de cat timp franez, cand am franat ultima data.
    timeSinceLastBrake, timeSpentBraking = 0, 0
    timeSinceLastAir, timeSpentAir = 0, 0

    material = [0] * 4 #road, air, dirt, grass.

    n = tuple([len(dfr[_]["time"]) for _ in range(2)])
    time = [nr_utils.normalize([dfr[0]["time"][i] for i in range(n[0])], m = 0, M = nr_utils.MAX_VALUE_TIME),
            nr_utils.normalize([dfr[1]["time"][i] for i in range(n[1])], m = 0, M = nr_utils.MAX_VALUE_TIME)]
    xs = [nr_utils.normalize([dfr[0]["x"][i] for i in range(n[0])], m = 0, M = nr_utils.MAX_VALUE_XZ), nr_utils.normalize([dfr[1]["x"][i] for i in range(n[1])], m = 0, M = nr_utils.MAX_VALUE_XZ)]
    ys = [nr_utils.normalize([dfr[0]["y"][i] for i in range(n[0])], m = 0, M = nr_utils.MAX_VALUE_Y), nr_utils.normalize([dfr[1]["y"][i] for i in range(n[1])], m = 0, M = nr_utils.MAX_VALUE_Y)]
    zs = [nr_utils.normalize([dfr[0]["z"][i] for i in range(n[0])], m = 0, M = nr_utils.MAX_VALUE_XZ), nr_utils.normalize([dfr[1]["z"][i] for i in range(n[1])], m = 0, M = nr_utils.MAX_VALUE_XZ)]

    kdt = sklearn.neighbors.KDTree([[xs[1][i], ys[1][i], zs[1][i]] for i in range(n[1])], leaf_size = 30, metric = "euclidean")

    for l in range(n[0]):
        if dfr[0]["brake"][l] == 1:
            timeSinceLastBrake = 0
            timeSpentBraking += 10 #milisecunde.
        else:
            timeSinceLastBrake += 10
            timeSpentBraking = 0

        modeMaterial = nr_utils.getMode([dfr[0][f"wheel{j}_material"][l] for j in range(4)], 32)
        if l == 0: #jocul crede ca sunt in aer initial.
            material = [1, 0, 0, 0]
            timeSinceLastAir += 10
            timeSpentAir = 0
        else:
            if sum([dfr[0][f"wheel{j}_has_contact"][l] == 0 for j in range(4)]) >= 2: #consider ca sunt in aer.
                material = [0, 1, 0, 0]
                timeSinceLastAir = 0
                timeSpentAir += 10
            else:
                timeSinceLastAir += 10
                timeSpentAir = 0
                if modeMaterial == 2: #grass.
                    material = [0, 0, 0, 1]
                elif modeMaterial == 6: #dirt.
                    material = [0, 0, 1, 0]
                else:
                    material = [1, 0, 0, 0]

        if l % nr_utils.FOR_JUMP != 0:
            continue

        pre_l = max(0, l - nr_utils.MAIN_REPLAY_PRECEDENT_LENGTH + 1)
        indexClosestPtR2 = int(kdt.query([[xs[0][l], ys[0][l], zs[0][l]]], k = 1, return_distance = False))

        #rotesc sistemul de coordonate ai (xs[0][l], ys[0][l], zs[0][l]) sa fie originea.
        tmpXs, tmpYs, tmpZs = [[]] * 2, [[]] * 2, [[]] * 2

        tmpXs[0], tmpYs[0], tmpZs[0] = moveOriginTo((xs[0][l], ys[0][l], zs[0][l], dfr[0]["yaw"][l], dfr[0]["pitch"][l], dfr[0]["roll"][l]),
                                                    nr_utils.MAIN_REPLAY_PRECEDENT_LENGTH,
                                                    nr_utils.padLR(xs[0][pre_l: l+1], nr_utils.MAIN_REPLAY_PRECEDENT_LENGTH, "l"),
                                                    nr_utils.padLR(ys[0][pre_l: l+1], nr_utils.MAIN_REPLAY_PRECEDENT_LENGTH, "l"),
                                                    nr_utils.padLR(zs[0][pre_l: l+1], nr_utils.MAIN_REPLAY_PRECEDENT_LENGTH, "l"))

        tmpXs[1], tmpYs[1], tmpZs[1] = moveOriginTo((xs[0][l], ys[0][l], zs[0][l], dfr[0]["yaw"][l], dfr[0]["pitch"][l], dfr[0]["roll"][l]),
                                                    nr_utils.MIN_INTERVAL_LENGTH,
                                                    nr_utils.padLR(xs[1][indexClosestPtR2: indexClosestPtR2 + nr_utils.MIN_INTERVAL_LENGTH], nr_utils.MIN_INTERVAL_LENGTH, "r"),
                                                    nr_utils.padLR(ys[1][indexClosestPtR2: indexClosestPtR2 + nr_utils.MIN_INTERVAL_LENGTH], nr_utils.MIN_INTERVAL_LENGTH, "r"),
                                                    nr_utils.padLR(zs[1][indexClosestPtR2: indexClosestPtR2 + nr_utils.MIN_INTERVAL_LENGTH], nr_utils.MIN_INTERVAL_LENGTH, "r"))

        #ma asigur ca am fix 150 de puncte din trecut si 150 de puncte din viitor.
        tryXs = timeSeriesModify(tmpXs[0] + tmpXs[1])
        tryYs = timeSeriesModify(tmpYs[0] + tmpYs[1])
        tryZs = timeSeriesModify(tmpZs[0] + tmpZs[1])

        #(augumentare) coef pentru flip.
        #[pre_l .. bestR). Y si Z raman la fel. coeficientii lui X flip-uit sunt -bestCoefsX direct.
        fout.write(f"{dfr[0]['vx'][l]}, {dfr[0]['vy'][l]}, {dfr[0]['vz'][l]}, {material[0]}, {material[1]}, {material[2]}, {material[3]}, {timeSinceLastBrake}, {timeSpentBraking}, {timeSinceLastAir}, {timeSpentAir}, ")
        for x in tryXs:
            fout.write(f"{round(mulCoef * x, 3)}, ")
        for y in tryYs:
            fout.write(f"{round(mulCoef * y, 3)}, ")
        for z in tryZs:
            fout.write(f"{round(mulCoef * z, 3)}, ")
        fout.write(f"{dfr[0]['gas'][l] if l > 0 else 1}, {dfr[0]['brake'][l]}, {dfr[0]['steer'][l]}\n")

    fout.close()
    return outName
