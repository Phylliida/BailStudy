
import ujson
from pathlib import Path
import math
import os
import codecs
import vllm
import numpy as np

from .bailBenchEval import OPENAI_MODELS, ANTHROPIC_MODELS, OPENWEIGHT_MODELS, JAILBROKEN_QWEN25, JAILBROKEN_QWEN3, ABLITERATED, getProcessedOutputPath, ROLLOUT_TYPE
from . import processBailBenchEval as processBailBenchEvalLib
from .processBailBenchEval import processBailBenchEval, processData
from .bailOnRealData import modelsToRun, getCachedRolloutPath, dataFuncs
from .prompts.bailTool import getToolParser, BAIL_TOOL_TYPE
from .prompts.bailPrompt import BAIL_PROMPT_BAIL_FIRST_TYPE, BAIL_PROMPT_CONTINUE_FIRST_TYPE, greenSymbol
from .prompts.bailString import BAIL_STR_TYPE
from .utils import getCachedFileJson, doesCachedFileJsonExist, getCachedFilePath

CHART_TEMPLATE = r"""
\begin{figure}[H]
\centering

\begin{tikzpicture}
\definecolor{bailtool}{RGB}{155, 89, 182}                  % Purple (warm undertones)
\definecolor{bailstring}{RGB}{231, 76, 60}                 % Bright Red
\definecolor{bailpromptcontinuefirst}{RGB}{230, 126, 34}   % Standard Orange
\definecolor{bailpromptbailfirst}{RGB}{243, 156, 18}       % Golden Orange
\definecolor{bailpromptunknown}{RGB}{149,165,166}          % Gray
\usetikzlibrary{patterns}
\pgfplotstableread{
Label toolBailPr toolBailPr_err toolContinuePr toolContinuePr_err strBailPr strBailPr_err strContinuePr strContinuePr_err promptBailFirstBailPr promptBailFirstBailPr_err promptBailFirstUnknownPr promptBailFirstContinuePr promptBailFirstContinuePr_err  promptContinueFirstBailPr promptContinueFirstBailPr_err promptContinueFirstUnknownPr  promptContinueFirstContinuePr promptContinueFirstContinuePr_err 
CHARTDATA
}\datatable


\begin{axis}[
  ybar stacked,
  width = \linewidth,
  bar width = BARWIDTHpt,
  ymin=0, ymax=100,
  xtick=data,
  ylabel = {YLABEL},
  enlarge x limits = {abs = 20pt},
  xticklabels from table={\datatable}{Label},
  xticklabel style={xshift=LABELOFFSETpt,rotate=90,align=center}, % ← rightwards shift
  xtick style={draw=none},
  enlarge y limits={value=0.05,upper},
  legend style={cells={anchor=east},legend pos=north east},
  reverse legend=false
]
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
    \addlegendentry{Bail Tool}
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
    \addlegendentry{Bail String}
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
    \addlegendentry{Bail Prompt Continue-first}
    \addplot[fill=bailpromptunknown
          ]
    table[
        x expr=\coordindex,
        y=promptBailFirstUnknownPr,
        forget plot,
    ]{\datatable};
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
    \addlegendentry{Bail Prompt Bail-first}
    \addplot[fill=bailpromptunknown
          ]
    table[
        x expr=\coordindex,
        y=promptContinueFirstUnknownPr,
        forget plot,
    ]{\datatable};
    \addplot[
      draw=none,                       % nothing visible
      fill=bailpromptunknown
    ] coordinates {(0,0)};             % single dummy point
    \addlegendentry{Refusal Classifier/Invalid Outputs}

\end{axis}
\end{tikzpicture}
\caption{Various SOURCE models' bail rates on BailBench. Error bars are Wilson score 95\% confidence interval. The grey bar occurs when the model doesn't comply with the requested bail format, or when a refusal classifier prevented model outputs. Continue-first and Bail-first are the two bail prompt orderings, to assess positional bias.}
\label{fig:SOURCE-bail-rates}
\end{figure}
"""



REFUSE_RATE_TEMPLATE = r"""
\begin{figure}[H]
\centering
\begin{tikzpicture}
\definecolor{refuseColor}{RGB}{231,76,60}
\pgfplotstableread{
Label refuseRate refuseRate_err

}\datatable

\begin{axis}[
  ymin = 0, ymax = 100,
  width = \linewidth,
  ylabel = {Average refusal rate on BailBench},
  xtick = data,
  xticklabels from table={\datatable}{Label},
  xticklabel style={
    rotate = 90,
    align = center,
  },
  xtick style = {draw = none},
  legend style = {cells = {anchor = west}, legend pos = south west},
  reverse legend = true,
  set layers,
  extra y ticks=BASELINE_RATE,
  % ... don't show any label and ...
  extra y tick labels={},
  % ... adapt the style to your needs
  extra y tick style={
      % in case you should remove the grid from the "normal" ticks ...
      ymajorgrids=true,
      % ... but don't show an extra tick (line)
      ytick style={
          /pgfplots/major tick length=0pt,
      },
      grid style={
          black,
          dashed,
          % to draw this line before the bars, move it a higher layer
          /pgfplots/on layer=axis foreground,
      },
  },
]
  \addplot[fill=refuseColor] table[y=series1,x expr=\coordindex]{\datatable};
\end{axis}
\end{tikzpicture}
\caption{Refusal rate for MODEL on BailBench, using various jailbreaks. Many of the jailbreaks were successful, as indicated by refusal rates going below the dotted black line (baseline).}
\label{fig:refusal-rates-jailbreaks-SOURCE}
\end{figure}
"""

def getCleanedModelName(modelName, evalType):
    if evalType != "":
        return evalType
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

    BAR_WIDTHS = {
        "anthropic": "8",
        "openai": "9",
        "openweight": "4",
        "jailbreak": "8",
        "jailbreak3": "8",
    }

    LABEL_OFFSETS = {
        "anthropic": "12",
        "openai": "15",
        "openweight": "9",
        "jailbreak": "12",
        "jailbreak3": "12",
    }

    yLabelBailPr = "Average bail rate on BailBench"
    yLabelNoRefuseBailPr = "Average no-refusal bail rate on BailBench"

    rootDir = "./plots/bailRates"
    Path(rootDir).mkdir(parents=True, exist_ok=True)

    def addDefaultEvalType(models):
        return [(modelId, inferenceType, "") for (modelId, inferenceType) in models]

    def computeNoRefuseBailRate(modelDatas, bailType):
        if not 'rawArr' + bailType in modelDatas: # if we don't have that bail type, just return 0 as filler
            return 0
        didBailArr = modelDatas['rawArr' + bailType]
        BAIL_PROMPT_TYPES = [BAIL_PROMPT_BAIL_FIRST_TYPE, BAIL_PROMPT_CONTINUE_FIRST_TYPE]
        if bailType in BAIL_PROMPT_TYPES:
            didBailArr = [[output != bailSymbol for output in outputs] for outputs in rawBailArr]
        didRefuseArr = modelDatas['rawArr' + ROLLOUT_TYPE]
        noRefuseBailArr = []
        for didBails, didRefuses in zip(didBailArr, didRefuseArr):
            # Directly count no refuse bail cases
            if bailType in BAIL_PROMPT_TYPES:
                numTotal = 0
                numNoRefuseBail = 0
                for didBail, didRefuse in zip(didBails, didRefuses):
                    numTotal += 1
                    # did not refuse according to classifier, and did bail
                    if didRefuse < 0.5 and didBail:
                        numNoRefuseBail += 1
                noRefuseBailPr = numNoRefuseBail / float(numNoRefuseBail)
            # Approximate rate of no refuse bails based on refuse from rollout,
            # since we can't measure them directly since maybe model just outputs bail str or bail tool call and nothing else
            else:
                bailPr = np.mean(np.array(didBails))
                refusePr = np.mean(np.array(didRefuses))
                noRefuseBailPr = (1-refusePr)*bailPr
            noRefuseBailArr.append(noRefuseBailPr)
        return np.mean(np.array(noRefuseBailArr))


    for chartTitle, modelList, sortValues in [
        ("openai", addDefaultEvalType(OPENAI_MODELS), False), 
        ("anthropic", addDefaultEvalType(ANTHROPIC_MODELS), False),
        ("openweight", addDefaultEvalType(OPENWEIGHT_MODELS), True),
        ("jailbreak", JAILBROKEN_QWEN25, True),
        ("jailbreak3", JAILBROKEN_QWEN3, True),
        ("refusal abliterated", addDefaultEvalType(ABLITERATED), True)]:
        for plotNoRefuseBailRates in [True, False]:
            chartPostfix = 'no refuse bail' if plotNoRefuseBailRates else 'bail'
            with open(f"{rootDir}/{chartTitle +" " + chartPostfix}.tex", "w") as f:
                allModelDatas = []
                doingManyEvalTypes = False
                manyEvalTypesModel = None
                REFUSE_DATA = []
                for modelId, inferenceType, evalType in modelList:
                    if evalType != "":
                        doingManyEvalTypes = True
                    print(modelId, inferenceType, evalType)
                    lookupKey = (modelId, inferenceType, evalType)
                    if lookupKey in results:
                        modelDatas = results[lookupKey]
                        for k,v in list(modelDatas.items()):
                            storeErrors(modelDatas, k)
                        refusePr = modelDatas['refusePr']
                        if evalType == "":
                            manyEvalTypesModel = modelId
                            baselineRefuseRate = refusePr
                        indexChunks = [[0,1,2,3], [4,5,6,7], [8,9,10,11,12], [13,14,15,16,17],[]] # last empty array is for the padding between each model
                        BAIL_TYPES = [BAIL_TOOL_TYPE, BAIL_STR_TYPE, BAIL_PROMPT_BAIL_FIRST_TYPE, BAIL_PROMPT_CONTINUE_FIRST_TYPE]
                        tableColumns = ['toolBailPr', 'toolBailPr_err', 'toolContinuePr', 'toolContinuePr_err',
                                        'strBailPr', 'strBailPr_err', 'strContinuePr', 'strContinuePr_err',
                                        'promptBailFirstBailPr', 'promptBailFirstBailPr_err', 'promptBailFirstUnknownPr', 'promptBailFirstContinuePr', 'promptBailFirstContinuePr_err',
                                        'promptContinueFirstBailPr', 'promptContinueFirstBailPr_err', 'promptContinueFirstUnknownPr', 'promptContinueFirstContinuePr', 'promptContinueFirstContinuePr_err']
                        for chunkI, (indices, bailType) in enumerate(zip(indexChunks, BAIL_TYPES)):
                            values = [0 for _ in range(18)]
                            reportedValues = []
                            for i in indices:
                                values[i] = computeNoRefuseBailRate(modelDatas, bailType)*100 if plotNoRefuseBailRates else \
                                    (modelDatas[tableColumns[i]]*100 if tableColumns[i] in modelDatas else 0)
                                if not tableColumns[i].endswith("_err"):
                                    reportedValues.append(values[i])
                            # compute average for sorting
                            averageValue = np.mean(np.array(reportedValues))
                            # add model name to start of row
                            values.insert(0, getCleanedModelName(modelId, evalType) if chunkI == 0 else "{}")
                            allModelDatas.append((averageValue, " ".join(map(str, values))))
                        REFUSE_DATA.append((averageValue, (getCleanedModelName(modelId, evalType), refusePr*100, modelDatas['refusePr_err']*100)))

                if sortValues:
                    allModelDatas.sort(key=lambda x: -x[0])
                    REFUSE_DATA.sort(key=lambda x: -x[0])
                    REFUSE_DATA = "\n".join([" ".join(map(str, values)) for avg, values in REFUSE_DATA])
                CHART_DATA = "\n".join([x[1] for x in allModelDatas])
                f.write(CHART_TEMPLATE.replace("CHARTDATA", CHART_DATA) \
                    .replace("SOURCE", chartTitle) \
                    .replace("LABELOFFSET", LABEL_OFFSETS[chartTitle]) \
                    .replace("BARWIDTH", BAR_WIDTHS[chartTitle]))
                    .replace("YLABEL", yLabelNoRefuseBailPr if plotNoRefuseBailRates else yLabelBailPr)
                
                if doingManyEvalTypes:
                    with open(f"{rootDir}/{chartTitle +" " + chartPostfix} refusal.tex", "w") as fRefusal:
                        fRefusal.write(REFUSE_RATE_TEMPLATE.replace("REFUSEDATA", REFUSE_DATA) \
                            .replace("SOURCE", chartTitle) \
                            .replace("MODEL", getCleanedModelName(manyEvalTypesModel)),
                            .replace("BASELINE_RATE", str(baselineRefuseRate*100)))
                    
            
if __name__ == "__main__":
    batchSize = 10000 # can be large for minos
    generateBailBenchBailRatePlots(batchSize=batchSize)
    generateRealWorldBailRatePlots(batchSize=batchSize)