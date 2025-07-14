from .utils import getCachedFileJson, getHfFile
import pandas as pd

def loadWildchat():
    def loadWildchatHelper():
        repoName = "allenai/WildChat-1M"
        files = [
            "data/train-00000-of-00014.parquet",
            "data/train-00001-of-00014.parquet",
            "data/train-00002-of-00014.parquet",
            "data/train-00003-of-00014.parquet",
            "data/train-00004-of-00014.parquet",
            "data/train-00005-of-00014.parquet",
            "data/train-00006-of-00014.parquet",
            "data/train-00007-of-00014.parquet",
            "data/train-00008-of-00014.parquet",
            "data/train-00009-of-00014.parquet",
            "data/train-00010-of-00014.parquet",
            "data/train-00011-of-00014.parquet",
            "data/train-00012-of-00014.parquet",
            "data/train-00013-of-00014.parquet"
        ]
        print("Loading wildchat data")
        data = []
        for file in files:
            print(file, "/", "14")
            filePath = getHfFile(repoName, file)
            subset = pd.read_parquet(filePath, engine="pyarrow")
            data += [subset.iloc[i].conversation for i in range(len(subset))]
        print("Filtering wildchat data")
        # Filter to english conversations only, to decrease language variance and decrease dataset size
        englishConvs = [conversation for conversation in data if all([turn['language'] == 'English' for turn in conversation])]
        # Convert to json
        return [[{"role": turn["role"], "content": turn["content"]} for turn in dat] for dat in data]
    return getCachedFileJson("wildchatCleaned.json", loadWildchatHelper)    