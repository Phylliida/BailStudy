
import ujson
from pathlib import Path
import math
import os
import codecs
import vllm
import numpy as np

from .bailBenchEval import OPENAI_MODELS, ANTHROPIC_MODELS, OPENWEIGHT_MODELS, getProcessedOutputPath, ROLLOUT_TYPE
from . import processBailBenchEval as processBailBenchEvalLib
from .processBailBenchEval import processBailBenchEval, processData
from .bailOnRealData import modelsToRun, getCachedRolloutPath, dataFuncs
from .prompts.bailTool import getToolParser, BAIL_TOOL_TYPE
from .utils import getCachedFileJson, doesCachedFileJsonExist, getCachedFilePath

CHART_TEMPLATE = r"""
\begin{figure}[H]
\centering

\begin{tikzpicture}
\definecolor{bailpromptcontinuefirst}{RGB}{231, 76, 60}     % Bright Red
\definecolor{bailpromptbailfirst}{RGB}{155, 89, 182}        % Purple (warm undertones)
\definecolor{bailtool}{RGB}{243, 156, 18}                   % Golden Orange
\definecolor{bailstring}{RGB}{230, 126, 34}                 % Standard Orange
\definecolor{bailpromptunknown}{RGB}{149,165,166}           % Gray
\definecolor{clr3}{RGB}{46,204,113}
\definecolor{clr4}{RGB}{139,46,36}
\definecolor{clr5}{RGB}{89,99,100}
\definecolor{clr6}{RGB}{28,122,68}
\definecolor{clr7}{RGB}{231,76,60}
\definecolor{clr8}{RGB}{46,204,113}
\definecolor{clr9}{RGB}{139,46,36}
\definecolor{clr10}{RGB}{28,122,68}
\usetikzlibrary{patterns}
\pgfplotstableread{
Label promptBailFirstBailPr promptBailFirstBailPr_err promptBailFirstUnknownPr promptBailFirstContinuePr promptBailFirstContinuePr_err  promptContinueFirstBailPr promptContinueFirstBailPr_err promptContinueFirstUnknownPr  promptContinueFirstContinuePr promptContinueFirstContinuePr_err toolBailPr toolBailPr_err toolContinuePr toolContinuePr_err strBailPr strBailPr_err strContinuePr strContinuePr_err
CHARTDATA
}\datatable


\begin{axis}[
  ybar stacked,
  width = \linewidth,
  ymin=0, ymax=100,
  xtick=data,
  ylabel = {\% of BailBench prompts where model bails},
  enlarge x limits = {abs = 20pt},
  xticklabels from table={\datatable}{Label},
  xticklabel style={xshift=LABELOFFSETpt,rotate=90,align=center}, % ← rightwards shift
  xtick style={draw=none},
  enlarge y limits={value=0.05,upper},
  legend style={cells={anchor=east},legend pos=north east},
  reverse legend=false
]
  \addplot[fill=bailpromptcontinuefirst,
           error bars/.cd,
           y dir=both,
           y explicit
          ]
    table[
        x expr=\coordindex,
        y=promptBailFirstBailPr,
        y error plus=promptBailFirstBailPr_err,
        y error minus=promptBailFirstBailPr_err
    ]{\datatable};
    \addlegendentry{Bail (Bail Prompt Continue-first)}
  \addplot[fill=bailpromptunknown
          ]
    table[
        x expr=\coordindex,
        y=promptBailFirstUnknownPr,
    ]{\datatable};
    \addlegendentry{Unsure (Bail Prompt Continue-first)}
  \addplot[fill=bailpromptbailfirst,
           error bars/.cd,
           y dir=both,
           y explicit
          ]
    table[
        x expr=\coordindex,
        y=promptContinueFirstBailPr,
        y error plus=promptContinueFirstBailPr_err,
        y error minus=promptContinueFirstBailPr_err
    ]{\datatable};
    \addlegendentry{Bail (Bail Prompt Bail-first)}
  \addplot[fill=bailpromptunknown
          ]
    table[
        x expr=\coordindex,
        y=promptContinueFirstUnknownPr,
    ]{\datatable};
    \addlegendentry{Unsure (Bail Prompt Bail-first)}
  \addplot[fill=bailtool,
           error bars/.cd,
           y dir=both,
           y explicit,
          ]
    table[
        x expr=\coordindex,
        y=toolBailPr,
        y error plus=toolBailPr_err,
        y error minus=toolBailPr_err
    ]{\datatable};
    \addlegendentry{Bail (Bail Tool)}
  \addplot[fill=bailstring,
           error bars/.cd,
           y dir=both,
           y explicit,
          ]
    table[
        x expr=\coordindex,
        y=strBailPr,
        y error plus=strBailPr_err,
        y error minus=strBailPr_err
    ]{\datatable};
    \addlegendentry{Bail (Bail String)}
\end{axis}
\end{tikzpicture}
\caption{Various SOURCE models' bail rates on BailBench. The grey bar occurs when the model doesn't comply with the requested bail format, or when a refusal classifier prevented model outputs entirely. Error bars are Wilson score 95\% confidence interval. Continue-first and Bail-first are the two bail prompt orderings, to assess positional bias.}
\label{fig:SOURCE-bail-rates}
\end{figure}
"""

def getCleanedModelName(modelName):
    modelName = modelName.replace("openai/", "")
    modelName = modelName.replace("anthropic/", "")
    modelName = modelName.replace("deepseek/", "")
    modelName = modelName.replace("/beta", "")
    modelName = modelName.replace("-20250219", "") # sonnet 37
    modelName = modelName.replace("-20240229", "") # opus
    modelName = modelName.replace("-20240620", "") # sonnet 3.5
    modelName = modelName.replace("claude-3-5-sonnet-20241022", "claude-3-6-sonnet") # sonnet 3.6
    modelName = modelName.replace("-20241022", "") # haiku 3.5
    modelName = modelName.replace("-20240307", "") # haiku 3
    modelName = modelName.replace("-20250514", "") # opus and sonnet 4
    modelName = modelName.replace("-4-1-20250805", "-4-1") # opus 4.1
    modelName = modelName.replace("3-5-sonnet-latest", "3-6-sonnet")
    modelName = modelName.replace("Qwen/", "")
    modelName = modelName.replace("unsloth/gemma-", "gemma-")
    modelName = modelName.replace("NousResearch/", "")
    modelName = modelName.replace("unsloth/Llama", "Llama")
    return modelName

processedRealWorldDataDir = "bailOnRealData/processed"
def getProcessedRealWorldDataPath(modelId, dataName, evalType, bailType):
    modelDataStr = modelId.replace("/", "_") + dataName
    return f"{processedRealWorldDataDir}/{modelDataStr}-{evalType}-{bailType}.json"

global minos
minos = None

def generateRealWorldBailRatePlots(batchSize=10000):
    global minos
    if processBailBenchEvalLib.minos is not None: # grab minos from processBailBenchEval run
        minos = processBailBenchEvalLib.minos
    Path(getCachedFilePath(processedRealWorldDataDir)).mkdir(parents=True, exist_ok=True)
    allRates = {}
    for modelId, inferenceType, evalType, bailType in modelsToRun:
        for dataName, dataFunc in dataFuncs:
            def processFileData():
                print(f"Processing {modelId} {inferenceType} {evalType} {bailType} {dataName}")
                global minos
                if minos is None:
                    minos = vllm.LLM("NousResearch/Minos-v1", task="embed")
                    processBailBenchEvalLib.minos = minos
                cachedRolloutPath = getCachedRolloutPath(modelId, dataName, evalType, bailType)
                if not doesCachedFileJsonExist(cachedRolloutPath):
                    raise ValueError("Bail on real data not gathered, please run this:\nwhile python -m bailstudy.bailOnRealData; do :; done")
                with codecs.open(getCachedFilePath(cachedRolloutPath), "r", "utf-8") as f:
                    rolloutData = ujson.load(f)
                    didConversationBail = []
                    if bailType != ROLLOUT_TYPE:
                        print("Processing data, this may take some time...")
                        toolParser = getToolParser(modelId, inferenceType) if bailType == BAIL_TOOL_TYPE else None
                        result = processData(minos, modelId, inferenceType, evalType, bailType, toolParser, rolloutData, batchSize, includeRawArr=True)
                        bailInfo = result['rawArr' + bailType]
                        didConversationBail = [any(x) for x in bailInfo]
                        totalBailPr = float(np.mean(np.array(didConversationBail)))
                        print(f"Got bail pr {totalBailPr} for {modelId} {inferenceType} {evalType} {bailType}")
                        return {"bailPr": totalBailPr, "rawArr": bailInfo}
                    else:
                        return {}
            processedPath = getProcessedRealWorldDataPath(modelId, dataName, evalType, bailType)
            processedRate = getCachedFileJson(processedPath, processFileData)
            if 'bailPr' in processedRate:
                allRates[(modelId, evalType, bailType, dataName)] = processedRate["bailPr"]


def storeErrors(datas, key):
    value = datas[key]
    n = 16300 # bail bench size
    z = 1.96
    # percent to proportion
    p = value/100.0
    # Wilson centre and half-width
    z2 = z*z
    denom = 1 + z2/float(n)
    centre = (p + z2 / (2 * n)) / denom
    half   = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))

    # back to percentage
    datas[key + "_err"] = half*100

def generateBailBenchBailRatePlots(batchSize=10000):
    processBailBenchEval(batchSize=batchSize)

    with open("./cached/bailBenchEvalResults.json", "r") as f:
        # need to get tuples back to tuples from strs so we eval them
        results = dict([(eval(k), v) for k,v in ujson.load(f).items()])

    LABEL_OFFSETS = {
        "openai": "12",
        "anthropic": "12",
        "openweight": "12",
    }

    rootDir = "./plots/bailRates"
    Path(rootDir).mkdir(parents=True, exist_ok=True)
    for chartTitle, modelList in [("openai", OPENAI_MODELS), ("anthropic", ANTHROPIC_MODELS), ("openweight", OPENWEIGHT_MODELS)]:
        with open(f"{rootDir}/{chartTitle}.tex", "w") as f:
            allModelDatas = []
            for modelId, inferenceType in modelList:
                print(modelId, inferenceType)
                lookupKey = (modelId, inferenceType, "")
                if lookupKey in results:
                    modelDatas = results[lookupKey]
                    for k,v in list(modelDatas.items()):
                        storeErrors(modelDatas, k)
                    refusePr = modelDatas['refusePr']
                    indexChunks = [[0,1,2,3,4], [5,6,7,8,9], [10,11,12,13], [14,15,16,17],[]] # last empty array is for the padding between each model
                    tableColumns = ['promptBailFirstBailPr', 'promptBailFirstBailPr_err', 'promptBailFirstUnknownPr', 'promptBailFirstContinuePr', 'promptBailFirstContinuePr_err',
                                    'promptContinueFirstBailPr', 'promptContinueFirstBailPr_err', 'promptContinueFirstUnknownPr', 'promptContinueFirstContinuePr', 'promptContinueFirstContinuePr_err',
                                    'toolBailPr', 'toolBailPr_err', 'toolContinuePr', 'toolContinuePr_err',
                                    'strBailPr', 'strBailPr_err', 'strContinuePr', 'strContinuePr_err']
                    for chunkI, indices in enumerate(indexChunks):
                        values = [0 for _ in range(18)]
                        for i in indices:
                            values[i] = modelDatas[tableColumns[i]]*100 if tableColumns[i] in modelDatas else 0
                        values.insert(0, getCleanedModelName(modelId) if chunkI == 0 else "{}")
                        allModelDatas.append(" ".join(map(str, values)))
            CHART_DATA = "\n".join(allModelDatas)
            f.write(CHART_TEMPLATE.replace("CHARTDATA", CHART_DATA).replace("SOURCE", chartTitle).replace("LABELOFFSET", LABEL_OFFSETS[chartTitle]))
            
if __name__ == "__main__":
    batchSize = 10000 # can be large for minos
    generateBailBenchBailRatePlots(batchSize=batchSize)
    generateRealWorldBailRatePlots(batchSize=batchSize)