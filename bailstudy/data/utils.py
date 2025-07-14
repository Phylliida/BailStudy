
import pathlib
import os
import ujson
import traceback
from huggingface_hub import hf_hub_download
from ..utils import getCachedFileJson

def getHfFile(repoId, fileName):
    return hf_hub_download(
        repo_id   = repoId,
        filename  = fileName,
        repo_type = "dataset",
        cache_dir = str(getCachedDir()),
    )