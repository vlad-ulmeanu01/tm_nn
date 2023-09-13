import pandas as pd
import sys
import os

import conv_make_input_from_csv_pair

cnt, finalCnt = 0, 3128
l, r = map(int, sys.argv[1:])

rawDataset = "/home/vlad/Desktop/Probleme/Trackmania/RawDataset/"
for path, subdirs, files in os.walk(rawDataset):
    #print(path, len(files))

    for i in range(len(files)):
        fPath1 = os.path.join(path, files[i])

        if pd.read_csv(fPath1, skipinitialspace = True)["steer"].nunique() <= 3:
            for j in range(len(files)):
                if j != i:
                    cnt += 1

                    if l <= cnt <= r:
                        fPath2 = os.path.join(path, files[j])
                        fName = conv_make_input_from_csv_pair.make_input_from_pair(fPath1, fPath2)

                        print(f"ok {cnt}/{finalCnt}: {fPath1}, {fPath2}, {fName}.")