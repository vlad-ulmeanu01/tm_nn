import pandas as pd
import torch
import copy

LEN_INPUT = 595 #139 #1626 #1820
LEN_INPUT_BEFORE_COEF = 13
LEN_INPUT_XYZ = 97 * 2 #x2 pentru +-.
LEN_OUTPUT_GAS = 2
LEN_OUTPUT_BRAKE = 2
LEN_OUTPUT_STEER = 3 #129

class SharedNet(torch.nn.Module):
    def __init__(self):
        super(SharedNet, self).__init__()

        self.stretchSigmoid = lambda x: torch.sigmoid(x / 4)  # torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU() #todo prelu?
        self.leakyRelu = torch.nn.LeakyReLU()
        self.drop = torch.nn.Dropout(0.2)

        self.bn2 = torch.nn.BatchNorm1d(512)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.bn4 = torch.nn.BatchNorm1d(128)
        self.bn5 = torch.nn.BatchNorm1d(64)
        self.bn6 = torch.nn.BatchNorm1d(32)

        self.fc1 = torch.nn.Linear(LEN_INPUT, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 256)
        self.fc4 = torch.nn.Linear(256, 128)
        self.fc5 = torch.nn.Linear(128, 64)
        self.fc6 = torch.nn.Linear(64, 32)
        self.fc7 = torch.nn.Linear(32, 16)
        self.fc8 = torch.nn.Linear(16, 8)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.drop(self.relu(self.bn2(self.fc2(out))))
        out = self.drop(self.relu(self.bn3(self.fc3(out))))
        out = self.drop(self.relu(self.bn4(self.fc4(out))))
        out = self.drop(self.relu(self.bn5(self.fc5(out))))
        out = self.drop(self.relu(self.bn6(self.fc6(out))))
        out = self.relu(self.fc7(out))
        out = self.relu(self.fc8(out))
        return out

class MainNet(torch.nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.sharedNet = SharedNet()

        self.softmax = torch.nn.Softmax(dim = -1)

        self.fc1_gas = torch.nn.Linear(8, LEN_OUTPUT_GAS)
        self.fc1_brake = torch.nn.Linear(8, LEN_OUTPUT_BRAKE)
        self.fc1_steer = torch.nn.Linear(8, LEN_OUTPUT_STEER)

    def forward(self, x):
        sharedOut = self.sharedNet(x)

        gasOut = self.fc1_gas(sharedOut)
        gasOut = self.softmax(gasOut)

        brakeOut = self.fc1_brake(sharedOut)
        brakeOut = self.softmax(brakeOut)

        steerOut = self.fc1_steer(sharedOut)
        steerOut = self.softmax(steerOut)

        return gasOut, brakeOut, steerOut

class SharedLoss(torch.nn.Module):
    def __init__(self):
        super(SharedLoss, self).__init__()
        self.loss_class = torch.nn.BCELoss()
        self.loss_steer = torch.nn.CrossEntropyLoss() #torch.nn.MSELoss()

    # presupun ca backward() se autocalculeaza dupa forward?
    def forward(self, yPred, yTruth):
        #!!y* este tuple!!!
        return self.loss_class(yPred[0], yTruth[0]) + self.loss_class(yPred[1], yTruth[1]) + self.loss_steer(yPred[2], yTruth[2])

class Dataset(torch.utils.data.Dataset):
    #Dataset cu 2n elemente (deocamdata fara aug).

    #Initialization.
    def __init__(self, dfr: pd.DataFrame, l, r):
        self.dfr = dfr
        self.l, self.r = l, r
        self.n_noaug = r - l + 1
        self.n = self.n_noaug * 2

    # Denotes the total number of samples.
    def __len__(self):
        return self.n

    # Generates one sample of data.
    def __getitem__(self, ind):
        self.is_aug = False if ind < self.n_noaug else True
        if self.is_aug:
            ind -= self.n_noaug

        xs = torch.FloatTensor(self.dfr.iloc[self.l + ind, 0: LEN_INPUT].values)

        arrGasValue = self.dfr.iloc[self.l + ind, LEN_INPUT: LEN_INPUT + 1].values[0]
        arrBrakeValue = self.dfr.iloc[self.l + ind, LEN_INPUT + 1: LEN_INPUT + 2].values[0]
        arrSteer = list(self.dfr.iloc[self.l + ind, LEN_INPUT + 2: LEN_INPUT + 2 + LEN_OUTPUT_STEER].values)

        if self.is_aug:
            # for i in range(LEN_INPUT_BEFORE_COEF, LEN_INPUT_BEFORE_COEF + LEN_INPUT_XYZ, 2): #x-1 pentru coef_x*
            #     aux = copy.deepcopy(xs[i])
            #     xs[i] = xs[i+1]
            #     xs[i+1] = aux

            st, en = LEN_INPUT_BEFORE_COEF, LEN_INPUT_BEFORE_COEF + LEN_INPUT_XYZ
            xs[st: en: 2], xs[st+1: en: 2] = xs[st+1: en: 2], xs[st: en: 2].clone()
            arrSteer = arrSteer[::-1] #flip output asteptat pentru steer.

        ys = torch.FloatTensor([arrGasValue, 1.0 - arrGasValue]),\
             torch.FloatTensor([arrBrakeValue, 1.0 - arrBrakeValue]),\
             torch.FloatTensor(arrSteer)

        return xs, ys