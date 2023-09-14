import pandas as pd

#incerc sa citesc intai dataframe-ul ca consuma ~3x pana cand mai da free-uri..
dfr = pd.read_csv("../MakeRefined/refined_kb_simple_conv_noaug.csv", skipinitialspace = True) #dfr = pd.read_csv("../MakeRefined/refined_kb_simple_conv_noaug.csv", skipinitialspace = True).sample(frac = 1).reset_index(drop = True) #random shuffle tot.
print("read dataframe.")

import numpy as np
import torch
import sys

import classes

net = classes.MainNet()
criterion = classes.SharedLoss()
optimizer = torch.optim.Adam(net.parameters())

#net.load_state_dict(torch.load("NetTM_partial.pt"))

n = len(dfr["v0"])
pc = 0.8

#TODO cand schimb datasetul: torch.utils.data.WeightedRandomSampler.

trainGen = torch.utils.data.DataLoader(classes.Dataset(dfr, 0, int(pc * n)), batch_size = 100, shuffle = True, num_workers = 2)
testGen = torch.utils.data.DataLoader(classes.Dataset(dfr, int(pc * n) + 1, n - 1), batch_size = 100)
print("ok gen!")

epochCnt = 100
trainLosses, testLosses = [], []
for epoch in range(epochCnt):  #Mini Batch gradient descent.
    totalLoss, fullBatchSizes = 0, 0

    for x, yTruth in trainGen:
        yPred = net(x)
        loss = criterion(yPred, yTruth)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        totalLoss += loss.item() * x.size()[0]  #lossVal.item() este deja "media".
        fullBatchSizes += x.size()[0]

        print(f"(partial) totalLoss = {totalLoss}, fullBatchSizes = {fullBatchSizes}")

    trainLosses.append(totalLoss / fullBatchSizes)

    print(f"Epoch: {epoch + 1}, Average Training Loss: {trainLosses[-1]}, ", end = '')

    with torch.set_grad_enabled(False):
        testLosses.append(0)

        totalCount = 0
        for x, yTruth in testGen:
            yPred = net(x)
            loss = criterion(yPred, yTruth)
            testLosses[-1] += loss.item() * x.size()[0]
            totalCount += x.size()[0]

        testLosses[-1] /= totalCount
        print(f"Average Test Loss: {testLosses[-1]}.")

    torch.save(net.state_dict(), f"NetTM_partial.pt")
    print(f"Saved NetTM_partial.pt.")

    if np.argmin(testLosses) == len(testLosses) - 1:
        torch.save(net.state_dict(), f"NetTM_best.pt")
        print(f"({epoch + 1}) Saved NetTM_best.pt")

    sys.stdout.flush()

print(f"trainLosses array = {trainLosses}")
print(f"testLosses array = {testLosses}")

#torch.save(net.state_dict(), "NetTM.pt")
#print(f"Saved NetTM.pt.")