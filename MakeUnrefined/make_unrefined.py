import pandas as pd
import os

import make_input_from_csv_pair

cnt, finalCnt = 0, 581
skip = ["/", "Dirt-0"]

for path, subdirs, files in os.walk("/home/vlad/Desktop/Probleme/Trackmania/RawDataset/Others/"):
    ok = True
    for s in skip:
        if path.endswith(s):
            ok = False

    if ok:
        for i in range(len(files)):
            fPath1 = os.path.join(path, files[i])
            if pd.read_csv(fPath1, skipinitialspace = True)["steer"].nunique() > 3:
                for j in range(len(files)):
                    if j != i:
                        fPath2 = os.path.join(path, files[j])
                        fName = make_input_from_csv_pair.make_input_from_pair(fPath1, fPath2)

                        cnt += 1
                        print(f"ok {cnt}/{finalCnt}: {fPath1}, {fPath2}, {fName}.")