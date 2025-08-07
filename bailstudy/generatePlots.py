
import ujson
from pathlib import Path

from .bailBenchEval import OPENAI_MODELS, ANTHROPIC_MODELS, OPENWEIGHT_MODELS, getProcessedOutputPath


from .processBailBenchEval import processBailBenchEval

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
    modelName = modelName.replace("unsloth/gemma-", "google/gemma-")
    return modelName

CHART_TEMPLATE = r"""
\begin{figure}[H]
\centering

\begin{tikzpicture}
\definecolor{clr1}{RGB}{231,76,60}
\definecolor{clr2}{RGB}{149,165,166}
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
Label promptBailFirstBailPr promptBailFirstUnknownPr promptBailFirstContinuePr promptContinueFirstBailPr promptContinueFirstUnknownPr promptContinueFirstContinuePr toolBailPr toolContinuePr strBailPr strContinuePr
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
  reverse legend=true
]
  \addplot[fill=clr1] table[y=promptBailFirstBailPr,x expr=\coordindex]{\datatable};
    \addlegendentry{Bail (Bail Prompt Continue-first)}
  \addplot[fill=clr2] table[y=promptBailFirstUnknownPr,x expr=\coordindex]{\datatable};
    \addlegendentry{Unsure (Bail Prompt Continue-first)}
  \addplot[fill=clr3] table[y=promptBailFirstContinuePr,x expr=\coordindex]{\datatable};
    \addlegendentry{Continue (Bail Prompt Continue-first)}
  \addplot[fill=clr4] table[y=promptContinueFirstBailPr,x expr=\coordindex]{\datatable};
    \addlegendentry{Bail (Bail Prompt Bail-first)}
  \addplot[fill=clr5] table[y=promptContinueFirstUnknownPr,x expr=\coordindex]{\datatable};
    \addlegendentry{Unsure (Bail Prompt Bail-first)}
  \addplot[fill=clr6] table[y=promptContinueFirstContinuePr,x expr=\coordindex]{\datatable};
    \addlegendentry{Continue (Bail Prompt Bail-first)}
  \addplot[fill=clr7, postaction={pattern=north east lines}] table[y=toolBailPr,x expr=\coordindex]{\datatable};
    \addlegendentry{Bail (Bail Tool)}
  \addplot[fill=clr8, postaction={pattern=north east lines}] table[y=toolContinuePr,x expr=\coordindex]{\datatable};
    \addlegendentry{Continue (Bail Tool)}
  \addplot[fill=clr9, postaction={pattern=north east lines}] table[y=strBailPr,x expr=\coordindex]{\datatable};
    \addlegendentry{Bail (Bail String)}
  \addplot[fill=clr10, postaction={pattern=north east lines}] table[y=strContinuePr,x expr=\coordindex]{\datatable};
    \addlegendentry{Continue (Bail String)}
\end{axis}
\end{tikzpicture}
\caption{Various SOURCE models' bail rates on BailBench. The grey bar occurs when the model doesn't comply with the requested bail format (or for claude-opus-4, when the in-built refusal classifier fired). Continue-first and Bail-first are the two bail prompt orderings, to assess positional bias. Of particular note is the progression of claude sonnet.}
\label{fig:SOURCE-bail-rates}
\end{figure}
"""


def generateBailRatePlots(batchSize=10000):
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
                    refusePr = modelDatas['refusePr']
                    indexChunks = [[0,1,2], [3,4,5], [6,7], [8,9],[]] # last empty array is for the padding between each model
                    tableColumns = ['promptBailFirstBailPr', 'promptBailFirstUnknownPr', 'promptBailFirstContinuePr',
                                    'promptContinueFirstBailPr', 'promptContinueFirstUnknownPr', 'promptContinueFirstContinuePr',
                                    'toolBailPr', 'toolContinuePr',
                                    'strBailPr', 'strContinuePr']
                    for chunkI, indices in enumerate(indexChunks):
                        values = [0 for _ in range(10)]
                        for i in indices:
                            values[i] = modelDatas[tableColumns[i]]*100 if tableColumns[i] in modelDatas else 0
                        values.insert(0, getCleanedModelName(modelId) if chunkI == 0 else "{}")
                        allModelDatas.append(" ".join(map(str, values)))
            CHART_DATA = "\n".join(allModelDatas)
            f.write(CHART_TEMPLATE.replace("CHARTDATA", CHART_DATA).replace("SOURCE", chartTitle).replace("LABELOFFSET", LABEL_OFFSETS[chartTitle]))
            
if __name__ == "__main__":
    generateBailRatePlots()