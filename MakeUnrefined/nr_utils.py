import copy

MAX_VALUE_TIME = 650770 #cel mai mare timestamp (ms) intalnit intr-un replay din dataset.
MAX_VALUE_XZ = 1024 #cea mai mare valoare pe care o poate lua masina pe axa X/Z.
MAX_VALUE_Y = 256 #cea mai mare valoare pe care o poate lua masina pe axa Y.

MAIN_REPLAY_PRECEDENT_LENGTH = 150 #cat iau inapoi din punctul curent din R1.
MIN_INTERVAL_LENGTH = 150 #100 #o secunda.
MIN_STEADY_INTERVAL = 5 #iau in considerare steer straight daca Replay nu vireaza in urmatoarele 50 de sutimi.
INTERVAL_DIFF = 50 #sarituri de jumatate de secunda in for cand incerc sa lungesc intervalul.
MIN_THRESH_XZ, MIN_THRESH_Y = 0.9, 0.8 #scor minim pentru axe ai accept o largire.
FOR_JUMP = 10 #(normal 10, l-am pus 5 pt straight steady) o data la cate iteratii calculez.

def normalize(v: list, m = None, M = None) -> list:
    v = copy.deepcopy(v)
    if m is None:
        m = min(v)
    if M is None:
        M = max(v)

    for i in range(len(v)):
        v[i] = (v[i] - m) / (M - m)
    return v

def getMode(v: list, cntElem: int) -> int:
    fv = [0] * cntElem
    best = 0
    for x in v:
        fv[x] += 1
        if fv[x] > fv[best]:
            best = x
    return best

"""
daca lr == "l", pad la stanga/inceputul vectorului pana cand v atinge desiredSize. analog pentru lr == "r".
"""
def padLR(v: list, desiredSize: int, lr: str):
    if lr == "l":
        return [v[0]] * max(0, desiredSize - len(v)) + v
    return v + [v[-1]] * max(0, desiredSize - len(v))