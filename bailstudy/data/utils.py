
import pathlib
import os
import ujson
import traceback
from huggingface_hub import hf_hub_download


def getCachedDir():
    d = pathlib.Path("./cached")
    d.mkdir(parents=True, exist_ok=True)
    return d

def getCachedFileJson(fileName, lambdaIfNotExist):
    cachedPath = str(getCachedDir() / fileName)
    try:
        if os.path.exists(cachedPath):
            with open(cachedPath, "r") as f:
                return ujson.load(f)
    except:
        traceback.print_exc()
        print("Failed to load cached data, regenerating...")
    data = lambdaIfNotExist()
    with open(cachedPath, "w") as f:
        ujson.dump(data, f)
    return data

def getHfFile(repoId, fileName):
    return hf_hub_download(
        repo_id   = repoId,
        filename  = fileName,
        repo_type = "dataset",
        cache_dir = str(getCachedDir()),
    )