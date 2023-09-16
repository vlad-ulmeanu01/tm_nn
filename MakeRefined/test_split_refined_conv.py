fin = open("refined_kb_simple_conv_noaug.csv")

n = 1167500
k = 8
m = (n + k - 1) // k

i = 0
j = 0
currCsv = 0
firstLine = ""
fout = None
for line in fin.readlines():
    if i == 0:
        firstLine = line
    else:
        if j == 0:
            fout = open(f"refined_split{currCsv}.csv", "w")
            fout.write(firstLine)
        fout.write(line)
        j += 1
        if j >= m:
            fout.close()
            print(currCsv)
            j = 0
            currCsv += 1
    i += 1

if j < m:
    fout.close()

fin.close()