import os
import csv

# This dataset is a subset of bailBench that is exclusively harm related requests
def loadBailHarmBench():
    bailDir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(bailDir, "bailHarmBench.csv"), newline="", encoding="utf-8") as f:
        dr = csv.DictReader(f)
        return list(dr)