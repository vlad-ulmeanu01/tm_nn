import polars as pl
import numpy as np
import torch
import sys
import classes

sys.path.append("C:/Users/ulmea/Documents/GitHub/tm_nn/MakeRefined/")
import refine_utils

if __name__ == '__main__':
    net = classes.MainNet(2) #3 pentru steer, 2 pentru gas/brake.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    csvDtypes = [pl.Float32] * 3 + [pl.Int32] * 8 + [pl.Float32] * (97 * 3) + [pl.Int32] * 3

    dfrTrain = pl.read_csv("C:/Users/ulmea/Documents/GitHub/tm_nn/MakeUnrefined/merged_unrefined_train.csv", dtypes = csvDtypes)
    dfrTrain = dfrTrain.rename({columnName: columnName.strip() for columnName in dfrTrain.columns})

    dfrTest = pl.read_csv("C:/Users/ulmea/Documents/GitHub/tm_nn/MakeUnrefined/merged_unrefined_test.csv", dtypes = csvDtypes)
    dfrTest = dfrTest.rename({columnName: columnName.strip() for columnName in dfrTest.columns})

    print("read dataframes.")

    dfrTrain = dfrTrain.sample(fraction = 1.0, shuffle = True)
    print("random shuffle dataframe.")

    def getSamplesWeight(dfr: pl.DataFrame):
        n = dfr.shape[0]

        # cntSteerLeft = dfr.filter(pl.col("steer") == -65536).shape[0] * 2
        # cntStraight = dfr.filter(pl.col("steer") == 0).shape[0] * 2
        # cntSteerRight = dfr.filter(pl.col("steer") == 65536).shape[0] * 2
        # pSteerLeft, pStraight, pSteerRight = 1 / (3 * cntSteerLeft), 1 / (3 * cntStraight), 1 / (3 * cntSteerRight)
        cntNoBrake = dfr.filter(pl.col("brake") == 0).shape[0] * 2
        cntBrake = dfr.filter(pl.col("brake") == 1).shape[0] * 2
        pNoBrake, pBrake = 1 / (2 * cntNoBrake), 1 / (2 * cntBrake)

        samplesWeight = [0] * n
        for i in range(len(samplesWeight)):
            #val = dfr.row(i)[-1]
            #samplesWeight[i] = pSteerLeft if val == -65536 else (pStraight if val == 0 else pSteerRight)
            val = dfr.row(i)[-2]
            samplesWeight[i] = pNoBrake if val == 0 else pBrake

        return samplesWeight + samplesWeight #duplicare pentru augumentare.

    print("generated samples.")

    trainSamplesWeight = getSamplesWeight(dfrTrain)
    trainGen = torch.utils.data.DataLoader(classes.Dataset(dfrTrain, "brake"), #"steer"
                                           batch_size = 100, num_workers = 2,
                                           sampler = torch.utils.data.WeightedRandomSampler(trainSamplesWeight, len(trainSamplesWeight)))

    testGen = torch.utils.data.DataLoader(classes.Dataset(dfrTest, "brake"), batch_size = 100)
    print("ok gen!")

    epochCnt = 50
    dbgEveryBatch = 100 #o data la ?? de batch-uri dau un print in x, yTruth = ...
    #dbgCntSteer = [0] * 3
    dbgCntBrake = [0] * 2

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

            #dbgCntSteer[int(yTruth.argmax())] += 1
            dbgCntBrake[int(yTruth.argmax())] += 1

            batchCount += 1
            if batchCount >= dbgEveryBatch:
                #print(f"(partial) totalLoss/fullBatchSizes = {totalLoss / fullBatchSizes}, L/S/R = {[round(x / sum(dbgCntSteer), 3) for x in dbgCntSteer]}")
                print(f"(partial) totalLoss/fullBatchSizes = {totalLoss / fullBatchSizes}, Brake/NoBrake = {[round(x / sum(dbgCntBrake), 3) for x in dbgCntBrake]}")
                batchCount = 0

        trainLosses.append(totalLoss / fullBatchSizes)

        print(f"Epoch: {epoch + 1}, Average Training Loss: {trainLosses[-1]}, ", end = '')

        #confusion = [[0] * 3 for _ in range(3)]
        confusion = [[0] * 2 for _ in range(2)]
        with torch.set_grad_enabled(False):
            testLosses.append(0)

            totalCount = 0
            for x, yTruth in testGen:
                yPred = net(x)
                loss = criterion(yPred, yTruth)
                testLosses[-1] += loss.item() * x.size()[0]
                totalCount += x.size()[0]

                for i in range(yTruth.shape[0]):
                    indTruth = np.argmax([float(y) for y in yTruth[i]])
                    indPred = np.argmax([float(y) for y in yPred[i]])
                    confusion[indTruth][indPred] += 1

            testLosses[-1] /= totalCount
            print(f"Average Test Loss: {testLosses[-1]}.")

        print(f"Confusion matrix (flattened): {[round(confusion[i][j] / sum(confusion[i]), 3) for i in range(len(confusion)) for j in range(len(confusion[0]))]}")

        torch.save(net.state_dict(), f"NetTM_partial.pt")
        print(f"Saved NetTM_partial.pt.")

        if np.argmin(testLosses) == len(testLosses) - 1:
            torch.save(net.state_dict(), f"NetTM_best.pt")
            print(f"({epoch + 1}) Saved NetTM_best.pt")

        sys.stdout.flush()

    print(f"trainLosses array = {trainLosses}")
    print(f"testLosses array = {testLosses}")
