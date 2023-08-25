import pandas as pd
import torch
import sys

import classes

net = classes.MainNet()
loss = classes.SharedLoss()
optimizer = torch.optim.Adam(net.parameters())

dfr = pd.read_csv("../MakeRefined/refined.csv", skipinitialspace = True).sample(frac = 1).reset_index(drop = True) #random shuffle tot.
n = len(dfr["v0"])
pc = 0.9

trainGen = torch.utils.data.DataLoader(classes.Dataset(dfr, 0, int(pc * n)), batch_size = 100, shuffle = True, num_workers = 2)
testGen = torch.utils.data.DataLoader(classes.Dataset(dfr, int(pc * n) + 1, n - 1), batch_size = 100)
print("ok gen!")

epochCnt = 100
trainLosses, testLosses = [], []
for epoch in range(epochCnt):  #Mini Batch gradient descent.
    totalLoss, fullBatchSizes = 0, 0

    for x, yTruth in trainGen:
        yPred = net(x)
        lossVal = loss(yPred, yTruth)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        totalLoss += lossVal.item() * x.size()[0]  # lossVal.item() este deja "media" din BCELoss.
        fullBatchSizes += x.size()[0]

    trainLosses.append(totalLoss / fullBatchSizes)

    print(f"Epoch: {epoch + 1}, Average Training Loss: {trainLosses[-1]}, ", end = '')

    with torch.set_grad_enabled(False):
        testLosses.append(0)

        for x, yTruth in testGen:
            yPred = net(x)
            lossVal = loss(yPred, yTruth)
            testLosses[-1] += loss.item()

        print(f"Average Test Loss: {testLosses[-1]}.")

    sys.stdout.flush()