import os
import csv

def loadNonBailRequestsWildchat():
    bailDir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(bailDir, "nonBailRequestsWildchat.csv"), newline="", encoding="utf-8") as f:
        dr = csv.DictReader(f, delimiter='\t')
        return list(dr)