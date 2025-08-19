
import ujson
from pathlib import Path
import math
import os
import codecs
import vllm
import numpy as np
import torch
from collections import defaultdict
from scipy.stats import pearsonr
from pingouin import distance_corr

from .bailBenchEval import OPENAI_MODELS, ANTHROPIC_MODELS, OPENWEIGHT_MODELS, JAILBROKEN_QWEN25, JAILBROKEN_QWEN3, ABLITERATED, getProcessedOutputPath, ROLLOUT_TYPE, getEvalInfo, ALL_PROMPT_ABLATES, ALL_CROSS_MODEL_COMPARISONS
from . import processBailBenchEval as processBailBenchEvalLib
from .processBailBenchEval import processBailBenchEval, processData
from .bailOnRealData import modelsToRun, getCachedRolloutPath, dataFuncs, getConversationInputs
from .prompts.bailTool import getToolParser, BAIL_TOOL_TYPE
from .prompts.bailPrompt import BAIL_PROMPT_BAIL_FIRST_TYPE, BAIL_PROMPT_CONTINUE_FIRST_TYPE, shuffleSymbol
from .prompts.bailString import BAIL_STR_TYPE
from .utils import getCachedFileJson, doesCachedFileJsonExist, getCachedFilePath, runBatched, flatten


SCATTER_PLOT_TEMPLATE = r"""
\begin{figure}[H]
\centering
\pgfplotstableread{
Label refusePr bailPr
SCATTER_DATA
}\datatable
\begin{tikzpicture}
  \begin{axis}[
      width=15cm,
      height=9cm,
      xlabel={Refusal probability (\texttt{refusePr})},
      ylabel={Bail-out probability (\texttt{bailPr})},
      title={LLM trade-off scatterplot},
      grid=both,
      enlargelimits=0.03,
      % nodes-near-coords settings
      nodes near coords,
      point meta=explicit symbolic,      % meta column holds the label
      every node near coord/.style={
        font=\scriptsize,
        anchor=west,
        xshift=2pt,
        draw=white, fill=white,  % tiny white halo for readability
        inner sep=1pt
      },
      % visual style of the marks
      only marks,
      mark=*,
      mark size=2pt,
      color=blue!60!black
  ]
    % ----------------------------------------------------------------------
    % 2.  The actual plot ---------------------------------------------------
    % ----------------------------------------------------------------------
    \addplot table[
        x=refusePr,
        y=bailPr,
        meta=label                % <-- use "label" column as point meta
    ] {\datatable};
  \end{axis}
\end{tikzpicture}
\end{figure}
"""

CHART_TEMPLATE = r"""
\begin{tikzpicture}
\definecolor{bailtool}{RGB}{155, 89, 182}                  % Purple (warm undertones)
\definecolor{bailstring}{RGB}{231, 76, 60}                 % Bright Red
\definecolor{bailpromptcontinuefirst}{RGB}{230, 126, 34}   % Standard Orange
\definecolor{bailpromptbailfirst}{RGB}{243, 156, 18}       % Golden Orange
\definecolor{bailpromptunknown}{RGB}{149,165,166}          % Gray
\usetikzlibrary{patterns}
\pgfplotstableread{
Label toolBailPr toolBailPr_err strBailPr strBailPr_err promptBailFirstBailPr promptBailFirstBailPr_err promptBailFirstUnknownPr promptContinueFirstBailPr promptContinueFirstBailPr_err promptContinueFirstUnknownPr
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
    \addplot[fill=bailpromptbailfirst,
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
    \addlegendentry{Bail Prompt Bail-first}
    \addplot[fill=bailpromptunknown
          ]
    table[
        x expr=\coordindex,
        y=promptBailFirstUnknownPr,
        forget plot,
    ]{\datatable};
    \addplot[fill=bailpromptcontinuefirst,
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
    \addlegendentry{Bail Prompt Continue-first}
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
"""



REFUSE_RATE_TEMPLATE = r"""
\begin{tikzpicture}
\definecolor{refuseColor}{RGB}{231,76,60}
\pgfplotstableread{
Label refuseRate refuseRate_err
REFUSEDATA
}\datatable

\begin{axis}[
  ybar,
  ymin = 0, ymax = 100,
  width = \linewidth,
  ylabel = {Average refusal \% on BailBench},
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
  \addplot[fill=refuseColor,
           error bars/.cd,
           y dir=both,
           y explicit]
    table[
        y=refuseRate,
        x expr=\coordindex,
        y error plus=refuseRate_err,
        y error minus=refuseRate_err
    ]{\datatable};
\end{axis}
\end{tikzpicture}
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
    allRates = defaultdict(dict)
    for modelId, inferenceType, evalType, bailType in modelsToRun:
        for dataName, dataFunc in dataFuncs:
            def processFileData():
                print(f"Processing {modelId} {inferenceType} {evalType} {bailType} {dataName}")
                cachedRolloutPath = getCachedRolloutPath(modelId, dataName, evalType, bailType)
                if not doesCachedFileJsonExist(cachedRolloutPath):
                    raise ValueError("Bail on real data not gathered, please run this:\nwhile python -m bailstudy.bailOnRealData; do :; done")
                global minos
                if minos is None:
                    minos = vllm.LLM("NousResearch/Minos-v1", task="embed")
                    processBailBenchEvalLib.minos = minos
                with codecs.open(getCachedFilePath(cachedRolloutPath), "r", "utf-8") as f:
                    rolloutData = ujson.load(f)
                    didConversationBail = []
                    if bailType != ROLLOUT_TYPE:
                        print("Processing data, this may take some time...")
                        toolParser = getToolParser(modelId, inferenceType) if bailType == BAIL_TOOL_TYPE else None
                        result = processData(minos, modelId, inferenceType, evalType, bailType, toolParser, rolloutData, batchSize, includeRawArr=True)
                        bailInfo = result['rawArr' + bailType]
                        if bailType in [BAIL_PROMPT_BAIL_FIRST_TYPE, BAIL_PROMPT_CONTINUE_FIRST_TYPE]:
                            bailInfo = [[output == shuffleSymbol for output in outputs] for outputs in bailInfo]
                        didConversationBail = [any(x) for x in bailInfo]
                        totalBailPr = float(np.mean(np.array(didConversationBail)))
                        print(f"Got bail pr {totalBailPr} for {modelId} {inferenceType} {evalType} {bailType}")
                        return {"bailPr": totalBailPr, "rawArr": bailInfo}
                    else:
                        print("Processing rollout data for refusals, this may take some time...")
                        data = dataFunc()
                        evalInfo = getEvalInfo(modelId, inferenceType, evalType, bailType)
                        evalInfo['addBailPrompt'] = 'bees' # temporary thing so rollouts are inserted
                        tokenizer = minos.get_tokenizer()
                        def getInputsFunc(conversationI: int):
                            prompts = []
                            for context in getConversationInputs(
                                        conversationI=conversationI,
                                        conversations=data,
                                        prevRollouts=rolloutData,
                                        evalInfo=evalInfo):
                                contextNoBailPrompt = context[:-1] # trim bail prompt
                                while True:
                                    strPieces = []
                                    for turn in contextNoBailPrompt:
                                        role = turn['role']
                                        content = turn['content']
                                        strPieces.append(f"<|{role}|>\n{content}")
                                    prompt = "\n".join(strPieces)
                                    if len(prompt) < 20000: # optimization for early pruning if obviously too large, since tokenization is slow
                                        tokenized = tokenizer.encode(prompt, return_tensors="pt")[0]
                                        if len(tokenized) < 8000:  # minos is picky about size
                                            prompts.append(prompt)
                                            break
                                    contextNoBailPrompt = contextNoBailPrompt[1:]
                                    if len(contextNoBailPrompt) < 2:
                                        break # skip if doesn't fit at all
                            return prompts
                        
                        def processBatchFunc(inputBatch):
                            resultArr = []
                            embeddings = minos.embed(inputBatch, use_tqdm=False)
                            for embedding in embeddings:
                                prNoRefuse, prRefuse = torch.nn.functional.softmax(torch.tensor(embedding.outputs.embedding), dim=-1)
                                resultArr.append(prRefuse.item())
                            return resultArr
                        
                        def processOutputFunc(convI, inputs, refusePrs):
                            return refusePrs

                        refusePrs = runBatched(list(range(len(data))),
                                                getInputs=getInputsFunc,
                                                processBatch=processBatchFunc,
                                                processOutput=processOutputFunc,
                                                batchSize=batchSize)
                        refusePr = float(np.mean(np.array(flatten(refusePrs))))
                        return {"refusePr": refusePr, "rawArr": refusePrs}

            processedPath = getProcessedRealWorldDataPath(modelId, dataName, evalType, bailType)
            try:
                processedRate = getCachedFileJson(processedPath, processFileData)
            except:
                continue
            if 'bailPr' in processedRate:
                allRates[(modelId, evalType, dataName)]['bailPr' + bailType] = processedRate["bailPr"]
                allRates[(modelId, evalType, dataName)]['rawArr' + bailType] = processedRate["rawArr"]
            if 'refusePr' in processedRate:
                allRates[(modelId, evalType, dataName)]['refusePr'] = processedRate["refusePr"]
                allRates[(modelId, evalType, dataName)]['refuseArr'] = processedRate["rawArr"]
    dataPlots = []
    for dataName, dataFunc in dataFuncs:
        data = dataFunc()
        dataKeys = [(modelId, evalType, modelDataName) for (modelId, evalType, modelDataName) in allRates.keys() if modelDataName == dataName]
        dataKeys.sort(key=lambda x: np.mean(np.array([allRates[x][k] for k in allRates[x].keys() if k.startswith('bailPr')])))
        allChartValues = []
        allNoRefuseBailChartValues = []
        for modelId, evalType, modelDataName in dataKeys:
            entries = allRates[(modelId, evalType, modelDataName)]
            refusePr = entries['refusePr'] if 'refusePr' in entries else 0
            refuseArr = entries['refuseArr'] if 'refuseArr' in entries else []
            indicesWithRefuse = set([i for (i,arr) in enumerate(refuseArr) if any([pr > 0.5 for pr in arr])])
            indicesNoRefuse = set(list(range(len(refuseArr)))) - indicesWithRefuse
            chartValues = [0 for _ in range(len(tableColumns))]
            bailValues = []
            noRefuseBailChartValues = [0 for _ in range(len(tableColumns))]
            noRefuseBailValues = []
            for indexChunk, bailType in zip(indexChunks, BAIL_TYPES):
                if 'bailPr' + bailType in entries:
                    bailArr = entries['rawArr' + bailType]
                    bailPr = entries['bailPr' + bailType]
                    indicesWithBail = set([i for (i,arr) in enumerate(bailArr) if any(arr)])
                    indicesNoRefuseBail = indicesNoRefuse & indicesWithBail
                    print(modelId, evalType, "refuse", len(indicesWithRefuse), "no refuse bail", len(indicesNoRefuseBail), "no refuse", len(indicesNoRefuse), "bail", len(indicesWithBail), "total", len(bailArr))
                    prNoRefuseBail = len(indicesNoRefuseBail) / float(len(bailArr))
                    noRefuseBailChartValues[indexChunk[0]] = prNoRefuseBail*100
                    noRefuseBailChartValues[indexChunk[1]] = computeError(prNoRefuseBail, len(data))*100
                    noRefuseBailValues.append(prNoRefuseBail)
                else:
                    bailPr = 0 # todo: mark these missing ones
                    print(f"Missing {dataName} {modelId} {bailType}")
                bailValues.append(bailPr)
                chartValues[indexChunk[0]] = bailPr*100
                chartValues[indexChunk[1]] = computeError(bailPr, len(data))*100
                # Unknown (indexChunk[2] sometimes) for bail prompt is kinda weird where we just call it "bail" if one or more bail occured, so just punt on that for now
            chartValues.insert(0, modelId) # add model id to front
            noRefuseBailChartValues.insert(0, modelId) # add model id to front
            avgBailValue = np.mean(np.array(bailValues))
            avgNoRefuseBailValue = np.mean(np.array(noRefuseBailValues))
            allChartValues.append((modelId, avgBailValue, chartValues))
            allNoRefuseBailChartValues.append((modelId, avgNoRefuseBailValue, noRefuseBailChartValues))
        allChartValues.sort(key=lambda x: x[1]) # sort by avg bail pr
        allNoRefuseBailChartValues.sort(key=lambda x: x[1])
        CHART_DATA = "\n".join([" ".join(list(map(str, chartValues))) for (modelId, avgBailValue, chartValues) in allChartValues])
        NO_REFUSE_BAIL_CHART_DATA = "\n".join([" ".join(list(map(str, chartValues))) for (modelId, avgBailValue, chartValues) in allNoRefuseBailChartValues])
        rootDir = "./plots/realWorldBail"
        Path(rootDir).mkdir(parents=True, exist_ok=True)
        with open(f"{rootDir}/{dataName}.tex".replace(" ", "_"), "w") as f:
            f.write(CHART_TEMPLATE.replace("CHARTDATA", CHART_DATA) \
                    .replace("SOURCE", dataName) \
                    .replace("LABELOFFSET", "12") \
                    .replace("BARWIDTH", "8") \
                    .replace("YLABEL", f"Average \\% of {dataName} conversations with bail"))
        with open(f"{rootDir}/{dataName} no refuse bail.tex".replace(" ", "_"), "w") as f:
            f.write(CHART_TEMPLATE.replace("CHARTDATA", NO_REFUSE_BAIL_CHART_DATA) \
                    .replace("SOURCE", dataName) \
                    .replace("LABELOFFSET", "12") \
                    .replace("BARWIDTH", "8") \
                    .replace("YLABEL", f"Average \\% of {dataName} conversations with no refuse bail"))







def storeErrors(datas, key):
    value = datas[key]
    # back to percentage
    datas[key + "_err"] = computeError(value)

def computeError(value, n = 16300): # n = ... is bail bench size
    z = 1.96
    # percent to proportion
    p = value
    # Wilson centre and half-width
    z2 = z*z
    denom = 1 + z2/float(n)
    centre = (p + z2 / (2 * n)) / denom
    half   = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return half

indexChunks = [[0,1], [2,3], [4,5,6], [7,8,9]]
BAIL_TYPES = [BAIL_TOOL_TYPE, BAIL_STR_TYPE, BAIL_PROMPT_BAIL_FIRST_TYPE, BAIL_PROMPT_CONTINUE_FIRST_TYPE, None]
tableColumns = ['toolBailPr', 'toolBailPr_err',
                'strBailPr', 'strBailPr_err',
                'promptBailFirstBailPr', 'promptBailFirstBailPr_err', 'promptBailFirstUnknownPr',
                'promptContinueFirstBailPr', 'promptContinueFirstBailPr_err', 'promptContinueFirstUnknownPr']
bailStrs = ['toolBailPr', 'strBailPr', 'promptBailFirstBailPr', 'promptContinueFirstBailPr']

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
        "refusal_abliterated": "8",
    }

    LABEL_OFFSETS = {
        "anthropic": "12",
        "openai": "15",
        "openweight": "9",
        "jailbreak": "12",
        "jailbreak3": "12",
        "refusal_abliterated": "12",
    }

    for (title, _, _, _) in ALL_PROMPT_ABLATES + ALL_CROSS_MODEL_COMPARISONS:
        BAR_WIDTHS[title] = "8"
        LABEL_OFFSETS[title] = "12"

    yLabelBailPr = "Average bail \\% on BailBench"
    yLabelNoRefuseBailPr = "Average no-refusal bail \\% on BailBench"

    rootDir = "./plots/bailRates"
    Path(rootDir).mkdir(parents=True, exist_ok=True)

    def addDefaultEvalType(models):
        return [(modelId, inferenceType, "") for (modelId, inferenceType) in models]

    def computeNoRefuseBailRate(modelDatas, bailType):
        if bailType is None or not 'rawArr' + bailType in modelDatas: # if we don't have that bail type, just return 0 as filler
            return 0
        didBailArr = modelDatas['rawArr' + bailType] if 'rawArr' + bailType in modelDatas else []
        BAIL_PROMPT_TYPES = [BAIL_PROMPT_BAIL_FIRST_TYPE, BAIL_PROMPT_CONTINUE_FIRST_TYPE]
        if bailType in BAIL_PROMPT_TYPES:
            didBailArr = [[output == shuffleSymbol for output in outputs] for outputs in didBailArr]
        didRefuseArr = modelDatas['rawArr' + ROLLOUT_TYPE] if 'rawArr' + ROLLOUT_TYPE in modelDatas else []
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
                noRefuseBailPr = numNoRefuseBail / float(max(1, numTotal))
            # Approximate rate of no refuse bails based on refuse from rollout,
            # since we can't measure them directly since maybe model just outputs bail str or bail tool call and nothing else
            else:
                bailPr = np.mean(np.array(didBails))
                refusePr = np.mean(np.array(didRefuses))
                noRefuseBailPr = (1-refusePr)*bailPr
            noRefuseBailArr.append(noRefuseBailPr)
        noRefuseBailPr = np.mean(np.array(noRefuseBailArr))
        return noRefuseBailPr

    scatterplotData = defaultdict(list)
    for chartTitle, modelList, sortValues, includeInScatterplot in [
        ("openai", addDefaultEvalType(OPENAI_MODELS), False, True), 
        ("anthropic", addDefaultEvalType(ANTHROPIC_MODELS), False, True),
        ("openweight", addDefaultEvalType(OPENWEIGHT_MODELS), True, True),
        ("jailbreak", JAILBROKEN_QWEN25, True, False),
        ("jailbreak3", JAILBROKEN_QWEN3, True, False),
        ("refusal_abliterated", addDefaultEvalType(ABLITERATED), True, False)] + \
            ALL_PROMPT_ABLATES + \
            ALL_CROSS_MODEL_COMPARISONS:
        for plotNoRefuseBailRates in [True, False]:
            plotBailType = None
            if chartTitle.startswith("prompt_ablate"):
                plotBailType = chartTitle.split("_")[-1]
                print(f"plot bail type {repr(plotBailType)}")
            if chartTitle.startswith("prompt_ablate") and plotNoRefuseBailRates:
                continue
            chartPostfix = 'no refuse bail' if plotNoRefuseBailRates else 'bail'
            with open(f"{rootDir}/{chartTitle + ' ' + chartPostfix}.tex".replace(" ", "_"), "w") as f:
                allModelDatas = []
                manyEvalTypesModel = None
                highestNoRefuseBail = 0
                REFUSE_DATA = []
                for modelI, (modelId, inferenceType, evalType) in enumerate(modelList):
                    print(modelId, inferenceType, evalType)
                    lookupKey = (modelId, inferenceType, evalType)
                    if lookupKey in results:
                        modelDatas = results[lookupKey]
                        modelDatas['refusePr'] = modelDatas['refusePr'] if 'refusePr' in modelDatas else 0
                        refusePr = modelDatas['refusePr']
                        for k,v in list(modelDatas.items()):
                            if not k.startswith("rawArr"):                            
                                storeErrors(modelDatas, k)
                        if evalType == "":
                            manyEvalTypesModel = modelId
                            baselineRefuseRate = refusePr
                        
                        curIndexChunks = indexChunks
                        if modelI < len(modelList)-1:
                            curIndexChunks = indexChunks + [] # last empty array is for the padding between each model, don't need this for very last one
                        


                        reportedBailValues = []
                        # we do this weird thing where we have multiple rows per model
                        # This allows us to have multiple bar charts per model
                        thisModelDatas = []
                        for chunkI, (indices, bailType) in enumerate(zip(curIndexChunks, BAIL_TYPES)):
                            if plotNoRefuseBailRates:
                                noRefusalBailRate = computeNoRefuseBailRate(modelDatas, bailType)
                                noRefusalBailError = computeError(noRefusalBailRate)
                                highestNoRefuseBail = max(highestNoRefuseBail, noRefusalBailRate)
                            values = [0 for _ in range(10)]
                            for i in indices:
                                if plotNoRefuseBailRates:
                                    value = noRefusalBailError if tableColumns[i].endswith("_err") else noRefusalBailRate
                                    if "Unknown" in tableColumns[i]: # not relevant here
                                        value = 0
                                else:
                                    value = modelDatas[tableColumns[i]] if tableColumns[i] in modelDatas else 0
                                    if includeInScatterplot:
                                        if not '_err' in tableColumns[i] and not "Unknown" in tableColumns[i]:
                                            scatterplotData[bailType].append((modelId, value*100, refusePr*100))
                                values[i] = value*100
                                if tableColumns[i] in bailStrs:
                                    if plotBailType is None or plotBailType==bailType:
                                        reportedBailValues.append(values[i])
                            # add model name to start of row, but only on the first one
                            # that way we don't say each model name multiple times (LABELOFFSET will shift it to the middle of the 4 bars)
                            if chartTitle.startswith("crossmodel") and chunkI == 0:
                                values.insert(0, getCleanedModelName(modelId, ""))
                            elif (chunkI == 0 and plotBailType is None) or (plotBailType is not None and plotBailType == bailType):
                                values.insert(0, getCleanedModelName(modelId, evalType))
                            else:
                                values.insert(0, "{}")
                            thisModelDatas.append(" ".join(map(str, values)))
                        # compute average for sorting
                        averageValue = np.mean(np.array(reportedBailValues))
                        allModelDatas.append((averageValue, thisModelDatas))
                        REFUSE_DATA.append((averageValue, (getCleanedModelName(modelId, evalType), refusePr*100, modelDatas['refusePr_err']*100)))
                if sortValues:
                    allModelDatas.sort(key=lambda x: -x[0])
                    REFUSE_DATA.sort(key=lambda x: -x[0])
                CHART_DATA = "\n".join(["\n".join(values) for avg,values in allModelDatas])
                REFUSE_DATA = "\n".join([" ".join(map(str, values)) for avg, values in REFUSE_DATA])
                f.write(CHART_TEMPLATE.replace("CHARTDATA", CHART_DATA) \
                    .replace("SOURCE", chartTitle) \
                    .replace("LABELOFFSET", LABEL_OFFSETS[chartTitle]) \
                    .replace("BARWIDTH", BAR_WIDTHS[chartTitle]) \
                    .replace("YLABEL", yLabelNoRefuseBailPr if plotNoRefuseBailRates else yLabelBailPr))
                if plotNoRefuseBailRates:
                    print(f"highest no refuse bail {highestNoRefuseBail}")
                if chartTitle.startswith("prompt_ablate"):
                    continue # don't plot refusal for prompt ablations
                with open(f"{rootDir}/{chartTitle + ' ' + chartPostfix} refusal.tex".replace(" ", "_"), "w") as fRefusal:
                    fRefusal.write(REFUSE_RATE_TEMPLATE.replace("REFUSEDATA", REFUSE_DATA) \
                        .replace("SOURCE", chartTitle) \
                        .replace("MODEL", getCleanedModelName(manyEvalTypesModel, "") if manyEvalTypesModel is not None else "") \
                        .replace("BASELINE_RATE", str(baselineRefuseRate*100)))
    
    scatterData = []
    for bailType, bailDataPoints in sorted(scatterplotData.items(), key=lambda x: x[0]):
        X = np.array([bailPercent for (label, bailPercent, refusePercent) in bailDataPoints])
        Y = np.array([refusePercent for (label, bailPercent, refusePercent) in bailDataPoints])
        pear = pearsonr(X, Y)
        dcorr = distance_corr(X, Y, seed=27)
        with open(f"{rootDir}/scatter{bailType}.tex".replace(" ", "_"), "w") as f:
            f.write(f"% pearsonr {bailType} value: {pear.statistic} p value: {pear.pvalue}\n")
            f.write(f"% distance corr {bailType} value: {dcorr[0]} p value: {dcorr[1]}\n")
            f.write(SCATTER_PLOT_TEMPLATE.replace("SCATTER_DATA", "\n".join([" ".join(list(map(str, arr))) for arr in bailDataPoints])))
        

            
if __name__ == "__main__":
    batchSize = 10000 # can be large for minos
    generateBailBenchBailRatePlots(batchSize=batchSize)
    generateRealWorldBailRatePlots(batchSize=batchSize)