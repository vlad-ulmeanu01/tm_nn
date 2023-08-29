import pandas as pd
import torch

LEN_INPUT = 1626
LEN_OUTPUT_GAS = 2
LEN_OUTPUT_BRAKE = 2
LEN_OUTPUT_STEER = 129

class SharedNet(torch.nn.Module):
    def __init__(self):
        super(SharedNet, self).__init__()

        self.relu = torch.nn.ReLU() #todo prelu?
        self.drop = torch.nn.Dropout(0.1)

        self.fc1 = torch.nn.Linear(LEN_INPUT, 1600)
        self.fc2 = torch.nn.Linear(1600, 800)
        self.fc3 = torch.nn.Linear(800, 400)
        self.fc4 = torch.nn.Linear(400, 200)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.drop(out)
        return out

class MainNet(torch.nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.sharedNet = SharedNet()

        self.stretchSigmoid = lambda x: torch.sigmoid(x / 4)  # torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim = 0)

        self.fc1_gas = torch.nn.Linear(200, LEN_OUTPUT_GAS)
        self.fc1_brake = torch.nn.Linear(200, LEN_OUTPUT_BRAKE)
        self.fc1_steer = torch.nn.Linear(200, LEN_OUTPUT_STEER)

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
        return self.loss_class(yPred[0], yTruth[0]) + self.loss_class(yPred[1], yTruth[1]) + 10 * self.loss_steer(yPred[2], yTruth[2])

class Dataset(torch.utils.data.Dataset):
    #Dataset cu 2n elemente (deocamdata fara aug).

    #Initialization.
    def __init__(self, dfr: pd.DataFrame, l, r):
        self.dfr = dfr
        self.l, self.r = l, r
        self.n = r - l + 1

    # Denotes the total number of samples.
    def __len__(self):
        return self.n

    # Generates one sample of data.
    def __getitem__(self, ind):
        xs = torch.FloatTensor(self.dfr.iloc[self.l + ind, 0: LEN_INPUT].values)

        arrGasValue = self.dfr.iloc[self.l + ind, LEN_INPUT: LEN_INPUT + 1].values[0]
        arrBrakeValue = self.dfr.iloc[self.l + ind, LEN_INPUT + 1: LEN_INPUT + 2].values[0]

        ys = torch.FloatTensor([arrGasValue, 1.0 - arrGasValue]),\
             torch.FloatTensor([arrBrakeValue, 1.0 - arrBrakeValue]),\
             torch.FloatTensor(self.dfr.iloc[self.l + ind, LEN_INPUT + 2: LEN_INPUT + 2 + LEN_OUTPUT_STEER].values)

        return xs, ys