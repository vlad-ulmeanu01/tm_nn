import pandas as pd
import sys
import os

import make_input_from_csv_pair

cnt, finalCnt = 0, 456
l, r = map(int, sys.argv[1:])

rawDataset = "C:/Users/ulmea/Desktop/Probleme/Trackmania/test_date_roti/"
for dir in [rawDataset + "Others/RL-Train1/", rawDataset + "Pro/"]:
    for path, subdirs, files in os.walk(dir):
        # ok = True
        # for s in ["/", "Dirt-0"]:
        #     if path.endswith(s):
        #         ok = False

        ok = (not path.endswith('/') or path.endswith("RL-Train1/"))

        if ok:
            for i in range(len(files)):
                fPath1 = os.path.join(path, files[i])
                if pd.read_csv(fPath1, skipinitialspace = True)["steer"].nunique() <= 3:
                    for j in range(len(files)):
                        if j != i:
                            cnt += 1

                            if l <= cnt <= r:
                                fPath2 = os.path.join(path, files[j])
                                fName = make_input_from_csv_pair.make_input_from_pair(fPath1, fPath2)

                                print(f"ok {cnt}/{finalCnt}: {fPath1}, {fPath2}, {fName}.")