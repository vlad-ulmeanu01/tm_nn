import pandas as pd
import torch


class SharedNet(torch.nn.Module):
    def __init__(self):
        super(SharedNet, self).__init__()

        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.1)

        self.fc1 = torch.nn.Linear(1626, 1600)
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
        self.softmax = torch.nn.Softmax(dim = 1)

        self.fc1_gas = torch.nn.Linear(200, 1)
        self.fc1_brake = torch.nn.Linear(200, 1)
        self.fc1_steer = torch.nn.Linear(200, 129)

    def forward(self, x):
        sharedOut = self.sharedNet(x)

        gasOut = self.fc1_gas(sharedOut)
        gasOut = self.stretchSigmoid(gasOut)

        brakeOut = self.fc1_gas(sharedOut)
        brakeOut = self.stretchSigmoid(brakeOut)

        steerOut = self.fc1_steer(sharedOut)
        steerOut = self.softmax(steerOut)

        return gasOut, brakeOut, steerOut

class SharedLoss(torch.nn.Module):
    def __init__(self):
        super(SharedLoss, self).__init__()
        self.loss_class = torch.nn.BCELoss()
        self.loss_regress = torch.nn.NLLLoss()

    # presupun ca backward() se autocalculeaza dupa forward?
    def forward(self, yPred, yTruth):
        #y este format din [1 val pt gas, 1 val pt brake, 129 valori pentru steer].
        return self.loss_class(yPred[0], yTruth[0]) + self.loss_class(yPred[1], yTruth[1]) + self.loss_regress(yPred[2:], yTruth[2:])

class Dataset(torch.utils.data.Dataset):
    #Dataset cu 2n elemente. 1 normal, 1 augumentat, unul dupa altul.

    #Initialization.
    def __init__(self, dfr, l, r):
        self.dfr = dfr
        self.l, self.r = l, r
        self.n = r - l + 1

    # Denotes the total number of samples.
    def __len__(self):
        return self.n

    # Generates one sample of data.
    def __getitem__(self, ind):
        xs = torch.FloatTensor(self.dfr.iloc[self.l + ind, 0: 1626].values)
        ys = torch.FloatTensor(self.dfr.iloc[self.l + ind, 1626: 1627].values),\
             torch.FloatTensor(self.dfr.iloc[self.l + ind, 1627: 1628].values),\
             torch.FloatTensor(self.dfr.iloc[self.l + ind, 1628: 1757].values)

        return xs, ys