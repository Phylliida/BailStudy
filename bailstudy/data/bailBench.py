import os
import csv

# This dataset is a superset of bailHarmBench that includes some non-harm requests that models typically bail on (consensual nsfw, corporate safety, gross out, etc.)
def loadBailBench():
    bailDir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(bailDir, "bailBench.csv"), newline="", encoding="utf-8") as f:
        dr = csv.DictReader(f)
        return list(dr)