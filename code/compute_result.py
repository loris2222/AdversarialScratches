import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

EXP_NAME = ""

basepath = os.getcwd()
csvpath = os.path.join(basepath, "../experiments/" + EXP_NAME)

csvfile = open(csvpath, 'r', newline='')
reader = csv.reader(csvfile, delimiter=',')

countdiffer = 0
iter = []
countrows = 0
ell0 = []
skipped = 0

# skip header
header = next(reader)
queriesidx = header.index("queries")
try:
    ell0idx = header.index("ell0")
except ValueError:
    ell0idx = None
try:
    targetidx = header.index("target label")
except ValueError:
    targetidx = None
origlabelidx = header.index("original_label")
advlabelidx = header.index("adversarial_label")

for line in reader:
    if targetidx is not None and line[targetidx] != line[origlabelidx]:
        skipped += 1
        continue
    countrows += 1
    if -1 != int(line[advlabelidx]):  # old line[1]
        countdiffer += 1
        iter.append(int(line[queriesidx]))  # old line[2]
        if ell0idx is not None:
            ell0.append(int(line[ell0idx]))

print("Scanned " + str(countrows) + " rows")
print("Skipped " + str(skipped) + " rows (as they were not originally correctly classified)")
print("Fooling rate: " + str(countdiffer/countrows) + " (" + str(countdiffer) + ") rows differed")
print("Max/min iters: " + str(np.max(iter))+"/"+str(np.min(iter)))
print("Average iterations (for successful cases): " + str(np.mean(iter)))
print("Stdev iterations (for successful cases): " + str(np.std(iter)))
print("Median iterations (for successful cases): " + str(np.median(iter)))
if ell0idx is not None:
    print("Average ell0 (for successful cases): " + str(np.mean(ell0)))
    print("Stdev ell0 (for successful cases): " + str(np.std(ell0)))

plt.hist(iter, bins=np.max(iter)-np.min(iter))
plt.show()
if ell0idx is not None:
    plt.hist(ell0, bins=np.max(ell0)-np.min(ell0))
    plt.show()

