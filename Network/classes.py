import polars as pl
import numpy as np
import torch
import copy
import sys


sys.path.append("C:/Users/ulmea/Documents/GitHub/tm_nn/MakeRefined/")
import refine_utils

LEN_INPUT = 599
LEN_INPUT_BEFORE_COEF = 17
LEN_INPUT_XYZ = 97 * 2
LEN_OUTPUT_GAS = 2
LEN_OUTPUT_BRAKE = 2
LEN_OUTPUT_STEER = 3

class MainNet(torch.nn.Module):
    def __init__(self, lenOutput: int):
        super(MainNet, self).__init__()

        self.stretchSigmoid = lambda x: torch.sigmoid(x / 4)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.2)
        self.softmax = torch.nn.Softmax(dim = -1)

        self.bn2 = torch.nn.BatchNorm1d(512)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.bn4 = torch.nn.BatchNorm1d(128)
        self.bn5 = torch.nn.BatchNorm1d(64)
        self.bn6 = torch.nn.BatchNorm1d(32)

        # self.fcdumb = torch.nn.Linear(2, lenOutput)
        self.fc1 = torch.nn.Linear(LEN_INPUT, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 256)
        self.fc4 = torch.nn.Linear(256, 128)
        self.fc5 = torch.nn.Linear(128, 64)
        self.fc6 = torch.nn.Linear(64, 32)
        self.fc7 = torch.nn.Linear(32, 16)
        self.fc8 = torch.nn.Linear(16, 8)
        self.fc9 = torch.nn.Linear(8, lenOutput)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.drop(self.relu(self.bn2(self.fc2(out))))
        out = self.drop(self.relu(self.bn3(self.fc3(out))))
        out = self.drop(self.relu(self.bn4(self.fc4(out))))
        out = self.drop(self.relu(self.bn5(self.fc5(out))))
        out = self.drop(self.relu(self.bn6(self.fc6(out))))
        out = self.relu(self.fc7(out))
        out = self.relu(self.fc8(out))
        out = self.relu(self.fc9(out))
        # out = self.relu(self.fcdumb(x))
        out = self.softmax(out)

        return out

class Dataset(torch.utils.data.Dataset):
    #Initialization.
    def __init__(self, dfr: pl.DataFrame, outputType: str):
        self.dfr = dfr
        self.n = dfr.shape[0] * 2
        self.ref = refine_utils.Refiner("C:/Users/ulmea/Documents/GitHub/tm_nn/MakeRefined/export_pts_conv_lg_raport.txt")
        self.outputType = outputType
        assert(outputType in ["gas", "brake", "steer"])

    #Denotes the total number of samples.
    def __len__(self):
        return self.n

    #Generates one sample of data.
    def __getitem__(self, ind):
        self.is_aug = False if ind < self.dfr.shape[0] else True
        if self.is_aug:
            ind -= self.dfr.shape[0]

        currRow = list(self.dfr.row(ind))

        xs = []

        xs.extend(refine_utils.refineSpeed(np.linalg.norm(np.array(currRow[:3])) * 3.6))
        xs.extend(currRow[3:7])
        xs.extend(self.ref.refineValue("timeSinceLastBrake", currRow[7]))
        xs.extend(self.ref.refineValue("timeSpentBraking", currRow[8]))
        xs.extend(self.ref.refineValue("timeSinceLastAir", currRow[9]))
        xs.extend(self.ref.refineValue("timeSpentAir", currRow[10]))
        z = 11
        for ch in ['x', 'y', 'z']:
            for i in range(97):
                if self.is_aug and ch == 'x':
                    xs.extend(self.ref.refineValue(ch, -currRow[z]))
                else:
                    xs.extend(self.ref.refineValue(ch, currRow[z]))
                z += 1

        # if self.is_aug:
        #     xs.extend(self.ref.refineValue('x', -currRow[11 + 80]))
        # else:
        #     xs.extend(self.ref.refineValue('x', currRow[11 + 80]))

        xs = torch.FloatTensor(xs)
        assert(currRow[-1] in [-65536, 0, 65536])

        if self.is_aug:
            currRow[z + 2] *= -1

        if self.outputType == "gas":
            ys = torch.FloatTensor([currRow[z], 1.0 - currRow[z]])
            #ys = torch.FloatTensor([1.0, 0.0])
        elif self.outputType == "brake":
            ys = torch.FloatTensor([currRow[z + 1], 1.0 - currRow[z + 1]])
            #ys = torch.FloatTensor([0.0, 1.0])
        else:
            ys = torch.FloatTensor([1, 0, 0] if currRow[-1] == -65536 else ([0, 1, 0] if currRow[-1] == 0 else [0, 0, 1]))

        return xs, ys