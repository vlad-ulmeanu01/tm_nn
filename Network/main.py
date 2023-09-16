import polars as pl
import numpy as np
import torch
import sys

import classes

if __name__ == '__main__':
    net = classes.MainNet()
    criterion = classes.SharedLoss()
    optimizer = torch.optim.Adam(net.parameters())

    #net.load_state_dict(torch.load("NetTM_partial.pt"))

    dfr = pl.read_csv("../MakeRefined/refined_split0.csv") #refined_kb_simple_conv_noaug
    dfr = dfr.rename({columnName: columnName.strip() for columnName in dfr.columns}) #skipinitialspace.
    print("read dataframe.")

    dfr = dfr.sample(fraction = 1.0, shuffle = True)
    print("random shuffle dataframe.")

    n = dfr.shape[0]
    pc = 0.75

    #TODO x2 pentru augumentare. limit ia primele ?? linii din dataframe.
    cntSteerLeft = dfr.limit(int(n * pc)).filter(pl.col("s_left") == 1).shape[0] #* 2
    cntStraight = dfr.limit(int(n * pc)).filter(pl.col("s_straight") == 1).shape[0]
    cntSteerRight = dfr.limit(int(n * pc)).filter(pl.col("s_right") == 1).shape[0] #* 2
    pSteerLeft, pStraight, pSteerRight = 1 / (3 * cntSteerLeft), 1 / (3 * cntStraight), 1 / (3 * cntSteerRight)

    samplesWeight = [0] * int(pc * n)
    for i in range(len(samplesWeight)):
        s = dfr.row(i)[-3:]
        samplesWeight[i] = pSteerLeft if s[0] == 1 else (pStraight if s[1] == 1 else pSteerRight)
    print("generated samples.")

    trainGen = torch.utils.data.DataLoader(classes.Dataset(dfr, 0, int(pc * n)),
                                           batch_size = 100, num_workers = 2,
                                           sampler = torch.utils.data.WeightedRandomSampler(samplesWeight, len(samplesWeight)))
    testGen = torch.utils.data.DataLoader(classes.Dataset(dfr, int(pc * n) + 1, n - 1), batch_size = 100)
    print("ok gen!")

    epochCnt = 50
    dbgEveryBatch = 100 #o data la 100 de batch-uri dau un print in x, yTruth = ...

    trainLosses, testLosses = [], []
    for epoch in range(epochCnt):  #Mini Batch gradient descent.
        totalLoss, fullBatchSizes = 0, 0

        batchCount = 0
        for x, yTruth in trainGen:
            yPred = net(x)
            loss = criterion(yPred, yTruth)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            totalLoss += loss.item() * x.size()[0]  #lossVal.item() este deja "media".
            fullBatchSizes += x.size()[0]

            batchCount += 1
            if batchCount >= dbgEveryBatch:
                print(f"(partial) totalLoss/fullBatchSizes = {totalLoss / fullBatchSizes}")
                batchCount = 0

        trainLosses.append(totalLoss / fullBatchSizes)

        print(f"Epoch: {epoch + 1}, Average Training Loss: {trainLosses[-1]}, ", end = '')

        confusion = [[0] * 3 for _ in range(3)]
        with torch.set_grad_enabled(False):
            testLosses.append(0)

            totalCount = 0
            for x, yTruth in testGen:
                yPred = net(x)
                loss = criterion(yPred, yTruth)
                testLosses[-1] += loss.item() * x.size()[0]
                totalCount += x.size()[0]

                for i in range(yTruth[2].shape[0]):
                    steerTruth = np.argmax([float(y) for y in yTruth[2][i]])
                    steerPred = np.argmax([float(y) for y in yPred[2][i]])
                    confusion[steerTruth][steerPred] += 1

            testLosses[-1] /= totalCount
            print(f"Average Test Loss: {testLosses[-1]}.")

        print(f"Confusion matrix (flattened): {[round(confusion[i][j] / sum(confusion[i]), 3) for i in range(3) for j in range(3)]}")

        torch.save(net.state_dict(), f"NetTM_partial.pt")
        print(f"Saved NetTM_partial.pt.")

        if np.argmin(testLosses) == len(testLosses) - 1:
            torch.save(net.state_dict(), f"NetTM_best.pt")
            print(f"({epoch + 1}) Saved NetTM_best.pt")

        sys.stdout.flush()

    print(f"trainLosses array = {trainLosses}")
    print(f"testLosses array = {testLosses}")
