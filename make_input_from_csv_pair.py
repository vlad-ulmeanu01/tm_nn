import sklearn.linear_model
import sklearn.neighbors
import pandas as pd
import numpy as np
import time as ty
import copy

import utils

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
intoarce [x**0, x**1, .., x**k, e**x, e**-x, 1 - e**x, 1 - e**-x, sin(x*pi), sin(x*pi + pi/6), .., sin(x*pi + 11pi/6)].
"""
def fitGetElements(x: float, k: int) -> list:
    sol, p = [], 1

    for i in range(k + 1):
        sol.append(p)
        p *= x

    sol.append(np.exp(-(1 - x) * 7.5))  # crestere inceata de la 0 la 1.
    sol.append(np.exp(-x * 7.5))  # scadere rapida de la 1 la 0.
    sol.append(1 - sol[-1])  # crestere rapida de la 0 la 1.
    sol.append(1 - sol[-3])  # scadere inceata de la 1 la 0.

    phase, diffPhase = 0, np.pi / 6
    for i in range(k + 5, k + 17):
        sol.append(np.sin(x * np.pi + phase))
        phase += diffPhase

    return sol

"""
primeste niste coeficienti si evalueaza punctul x in functie de ei. mai este transmis si un intreg k (gradul maxim al
polinomului care face parte din coeficienti).
"""
def fitEvaluate(coefs: list, x: float, k: int) -> float:
    return float(np.dot(coefs, fitGetElements(x, k)))

"""
primesc un grafic y(x). vreau fit (poly pana la gradul k) pe x, y.
intorc lista de coeficienti si eroarea obtinuta la fit.
exemplele de la inceput sa aiba greutate mai mare.
modelul are penalizare pentru complexitate mare (regularizare).
"""
def fitInterval(x: list, y: list, k: int) -> (list, float):
    x, y = copy.deepcopy(x), copy.deepcopy(y)
    mulx, muly = 1 / max(x) if abs(max(x)) > 1e-9 else 1, 1 / max(y) if abs(max(y)) > 1e-9 else 1

    model = sklearn.linear_model.Ridge(alpha = 1e-10, fit_intercept = False) #model regresie liniara cu regularizare.

    X = np.zeros((len(x), k+17))
    for i in range(len(x)):
        j = 0
        for elem in fitGetElements(x[i], k):
            X[i][j] = elem * mulx
            j += 1

    Y = np.array(y) * muly
    sampleWeights = [np.exp(-i * 0.0078125) for i in range(len(x))]

    model.fit(X, Y, sampleWeights)
    return model.coef_ * (mulx / muly), model.score(X, Y, sampleWeights)  #vreau ca coeficientul dominant sa fie primul.

"""
primeste 2 nume de fisiere csv. intoarce numele csv-ului in care au fost scrise niste inputuri nerafinate pentru retea.
"""
def make_input_from_pair(fName1: str, fName2: str) -> str:
    dfr = [pd.read_csv(fName1, skipinitialspace = True), pd.read_csv(fName2, skipinitialspace = True)]

    outName = str(int(ty.time() * 10000)) + ".csv"
    fout = open(outName, "w")
    fout.write(f"vx, vy, vz, road, air, dirt, grass, timeSinceLastBrake, timeSpentBraking, timeSinceLastAir, timeSpentAir, " +
               ''.join(f"coef_x{i}, " for i in range(21)) + ''.join(f"coef_y{i}, " for i in range(21)) + ''.join(f"coef_z{i}, " for i in range(21)) +
               "gas, brake, steer" + "\n")

    #de cat timp franez, cand am franat ultima data.
    timeSinceLastBrake, timeSpentBraking = 0, 0
    timeSinceLastAir, timeSpentAir = 0, 0

    material = [0] * 4 #road, air, dirt, grass.

    n = tuple([len(dfr[_]["time"]) for _ in range(2)])
    time = [utils.normalize([dfr[0]["time"][i] for i in range(n[0])], m = 0, M = utils.MAX_VALUE_TIME),
            utils.normalize([dfr[1]["time"][i] for i in range(n[1])], m = 0, M = utils.MAX_VALUE_TIME)]
    xs = [utils.normalize([dfr[0]["x"][i] for i in range(n[0])], m = 0, M = utils.MAX_VALUE_XZ), utils.normalize([dfr[1]["x"][i] for i in range(n[1])], m = 0, M = utils.MAX_VALUE_XZ)]
    ys = [utils.normalize([dfr[0]["y"][i] for i in range(n[0])], m = 0, M = utils.MAX_VALUE_Y), utils.normalize([dfr[1]["y"][i] for i in range(n[1])], m = 0, M = utils.MAX_VALUE_Y)]
    zs = [utils.normalize([dfr[0]["z"][i] for i in range(n[0])], m = 0, M = utils.MAX_VALUE_XZ), utils.normalize([dfr[1]["z"][i] for i in range(n[1])], m = 0, M = utils.MAX_VALUE_XZ)]

    kdt = sklearn.neighbors.KDTree([[xs[1][i], ys[1][i], zs[1][i]] for i in range(n[1])], leaf_size = 30, metric = "euclidean")

    timeCuanta = time[0][1] - time[0][0]
    for l in range(n[0]):
        if dfr[0]["brake"][l] == 1:
            timeSinceLastBrake = 0
            timeSpentBraking += 10 #milisecunde.
        else:
            timeSinceLastBrake += 10
            timeSpentBraking = 0

        modeMaterial = utils.getMode([dfr[0][f"wheel{j}_material"][l] for j in range(4)], 32)
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

        if l % utils.FOR_JUMP != 0:
            continue

        pre_l = max(0, l - utils.MAIN_REPLAY_PRECEDENT_LENGTH + 1)
        indexClosestPtR2 = int(kdt.query([[xs[0][l], ys[0][l], zs[0][l]]], k = 1, return_distance = False))

        #rotesc sistemul de coordonate ai (xs[0][l], ys[0][l], zs[0][l]) sa fie originea.
        tmpXs, tmpYs, tmpZs = [None] * 2, [None] * 2, [None] * 2

        tmpXs[0], tmpYs[0], tmpZs[0] = moveOriginTo((xs[0][l], ys[0][l], zs[0][l], dfr[0]["yaw"][l], dfr[0]["pitch"][l], dfr[0]["roll"][l]),
                                                    n[0], xs[0], ys[0], zs[0])
        tmpXs[1], tmpYs[1], tmpZs[1] = moveOriginTo((xs[0][l], ys[0][l], zs[0][l], dfr[0]["yaw"][l], dfr[0]["pitch"][l], dfr[0]["roll"][l]),
                                                    n[1], xs[1], ys[1], zs[1])

        #trebuie sa am grija sa fie timpii consecutivi (tb shiftati cei din R2 sa inceapa fix dupa time[0][l]).
        nextTime = 0

        #vreau ca partea de timp "pre-" sa se termine fix inainte de 0.
        tryTime = list(np.array(time[0][pre_l : l+1]) - (time[0][l] + timeCuanta)) +\
                  list(np.array(time[1][indexClosestPtR2 : indexClosestPtR2 + utils.MIN_INTERVAL_LENGTH]) + (nextTime - time[1][indexClosestPtR2]))

        tryXs = tmpXs[0][pre_l : l+1] + tmpXs[1][indexClosestPtR2 : indexClosestPtR2 + utils.MIN_INTERVAL_LENGTH]
        tryYs = tmpYs[0][pre_l : l+1] + tmpYs[1][indexClosestPtR2 : indexClosestPtR2 + utils.MIN_INTERVAL_LENGTH]
        tryZs = tmpZs[0][pre_l : l+1] + tmpZs[1][indexClosestPtR2 : indexClosestPtR2 + utils.MIN_INTERVAL_LENGTH]

        bestCoefsX, bestFitX = fitInterval(tryTime, tryXs, 4)
        bestCoefsY, bestFitY = fitInterval(tryTime, tryYs, 4)
        bestCoefsZ, bestFitZ = fitInterval(tryTime, tryZs, 4)
        bestR = indexClosestPtR2 + utils.MIN_INTERVAL_LENGTH

        nowFitX, nowFitY, nowFitZ, r = bestFitX, bestFitY, bestFitZ, indexClosestPtR2 + utils.MIN_INTERVAL_LENGTH
        minCntTries = 3 #las minim 3 incercari.
        while r < n[1] and (minCntTries > 0 or (nowFitX > utils.MIN_THRESH_XZ and nowFitY > utils.MIN_THRESH_Y and nowFitZ > utils.MIN_THRESH_XZ)):
            minCntTries -= 1
            nextTime = tryTime[-1] + timeCuanta

            tryTime.extend(list(np.array(time[1][r : r + utils.INTERVAL_DIFF]) + (nextTime - time[1][r])))
            tryXs.extend(tmpXs[1][r : r + utils.INTERVAL_DIFF])
            tryYs.extend(tmpYs[1][r : r + utils.INTERVAL_DIFF])
            tryZs.extend(tmpZs[1][r : r + utils.INTERVAL_DIFF])

            r = min(n[1], r + utils.INTERVAL_DIFF)

            nowCoefsX, nowFitX = fitInterval(tryTime, tryXs, 4)
            nowCoefsY, nowFitY = fitInterval(tryTime, tryYs, 4)
            nowCoefsZ, nowFitZ = fitInterval(tryTime, tryZs, 4)

            if (nowFitX > utils.MIN_THRESH_XZ and nowFitY > utils.MIN_THRESH_Y and nowFitZ > utils.MIN_THRESH_XZ) or\
                (nowFitX > bestFitX and nowFitY > bestFitY and nowFitZ > bestFitZ):
                bestCoefsX, bestFitX = nowCoefsX, nowFitX
                bestCoefsY, bestFitY = nowCoefsY, nowFitY
                bestCoefsZ, bestFitZ = nowCoefsZ, nowFitZ
                bestR = r

        #(augumentare) coef pentru flip.
        #[pre_l .. bestR). Y si Z raman la fel. coeficientii lui X flip-uit sunt -bestCoefsX direct.
        fout.write(f"{dfr[0]['vx'][l]}, {dfr[0]['vy'][l]}, {dfr[0]['vz'][l]}, {material[0]}, {material[1]}, {material[2]}, {material[3]}, {timeSinceLastBrake}, {timeSpentBraking}, {timeSinceLastAir}, {timeSpentAir}, ")
        for i in range(21):
            fout.write(f"{round(bestCoefsX[i], 3)}, ")
        for i in range(21):
            fout.write(f"{round(bestCoefsY[i], 3)}, ")
        for i in range(21):
            fout.write(f"{round(bestCoefsZ[i], 3)}, ")
        fout.write(f"{dfr[0]['gas'][l] if l > 0 else 1}, {dfr[0]['brake'][l]}, {dfr[0]['steer'][l]}\n")

    fout.close()
    return outName
