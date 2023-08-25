import numpy as np

neg_exp_table = [np.exp(x) for x in np.linspace(-100, 0, 10**6)]
def getNegExp(x):
    return 0.0 if x < -100 else neg_exp_table[max(0, min(len(neg_exp_table) - 1, int(x * 1e4 + 1000000)))]

"""
Regula Freedman–Diaconis.
imi iau niste puncte de interes din interval. pentru fiecare punct din v, vad cat de departe este de fiecare punct ales.
"""
def refineValues(v: list, m = None, M = None, cntIntervals = None, maxDistToPoint = None):
    binWidth = None
    if m is None:
        m = min(v)
    if M is None:
        M = max(v)

    if cntIntervals is None:
        q1, q3 = np.percentile(v, [25, 75])
        binWidth = 200 * (q3 - q1) / np.cbrt(len(v)) #2x era normal.
        if binWidth < 1e-9:
            binWidth = M - m
        cntIntervals = min(100, max(1, int(np.ceil((M - m) / binWidth))))

    pts = np.linspace(m, M, cntIntervals + 1)

    #distanta maxima de care imi pasa. daca un punct este fix intre 2 intervale, nu as vrea sa am exponentul mai mic de -1.
    if maxDistToPoint is None:
        maxDistToPoint = 0.5 * binWidth

    return [[getNegExp(-((x - p) / maxDistToPoint) ** 2) for p in pts] for x in v]

"""
vreau viteza x in km/h.
"""
def refineSpeed(x: float):
    shifts = [100, 162, 235, 342, 501]
    x = max(min(500.0, x), 0.0)

    #0 -- 100 -- 162 -- 235 -- 342 -- 500.
    sol = [0.0] * 5
    i = 0
    while i < 5 and x > shifts[i]:
        sol[i] = 1
        i += 1

    sol[i] = x / shifts[i] if i == 0 else (x - shifts[i-1]) / (shifts[i] - shifts[i-1])
    return sol

CNT_INTERVALS_STEER = 128
MIN_STEER, MAX_STEER = -65536, 65536
STEER_MAX_DIST_TO_POINT = 512 #1024 / 2.
STEER_PTS = np.linspace(MIN_STEER, MAX_STEER, CNT_INTERVALS_STEER + 1)

"""
primesc un output cu CNT_INTERVALS_STEER pozitii (tipic output-ul retelei). determin care valoare de steer este cea mai
probabila sa fi produs output-ul. 
"""
def reverseGetSteer(output):
    ind = np.argmax(output)
    diffP = np.sqrt(-np.log(output[ind])) * STEER_MAX_DIST_TO_POINT
    xPosib = np.array([-diffP, diffP]) + min(MAX_STEER, MIN_STEER + ind * (2 * STEER_MAX_DIST_TO_POINT))

    errs = [np.linalg.norm(np.array(refineValues(
        [x], m = MIN_STEER, M = MAX_STEER, cntIntervals = CNT_INTERVALS_STEER, maxDistToPoint = STEER_MAX_DIST_TO_POINT
    )[0]) - np.array(output)) for x in xPosib]
    return xPosib[np.argmin(errs)]

