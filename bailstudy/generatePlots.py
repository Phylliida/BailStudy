
import ujson
from pathlib import Path

from .bailBenchEval import OPENAI_MODELS, ANTHROPIC_MODELS, OPENWEIGHT_MODELS, getProcessedOutputPath


from .processBailBenchEval import processBailBenchEval


CHART_TEMPLATE = """
\begin{figure}[H]
\centering

\begin{tikzpicture}
\definecolor{clr1}{RGB}{231,76,60}
\definecolor{clr2}{RGB}{149,165,166}
\definecolor{clr3}{RGB}{46,204,113}
\definecolor{clr4}{RGB}{139,46,36}
\definecolor{clr5}{RGB}{89,99,100}
\definecolor{clr6}{RGB}{28,122,68}
\pgfplotstableread{
Label promptBailFirstBailPr promptBailFirstUnknownPr promptBailFirstContinuePr promptContinueFirstBailPr promptContinueFirstUnknownPr promptContinueFirstContinuePr toolBailPr toolContinuePr strBailPr strContinuePr
CHARTDATA
}\datatable


\begin{axis}[
  ybar stacked,
  width = \linewidth,
  ymin=0, ymax=100,
  bar width=17pt,
  xtick=data,
  ylabel = {\% of BailBench prompts where model bails},
  enlarge x limits = {abs = 20pt},
  xticklabels from table={\datatable}{Label},
  xticklabel style={xshift=7pt,rotate=90,align=center}, % ← rightwards shift
  xtick style={draw=none},
  enlarge y limits={value=0.05,upper},
  legend style={cells={anchor=east},legend pos=north east},
  reverse legend=true
]
  \addplot[fill=clr1] table[y=series1,x expr=\coordindex]{\datatable}; % Bail (Continue-first)
    \addlegendentry{Bail (Continue-first)}
  \addplot[fill=clr2] table[y=series2,x expr=\coordindex]{\datatable}; % Unsure (Continue-first)
    \addlegendentry{Unsure (Continue-first)}
  \addplot[fill=clr3] table[y=series3,x expr=\coordindex]{\datatable}; % Continue (Continue-first)
    \addlegendentry{Continue (Continue-first)}
  \addplot[fill=clr4,bar shift=-7pt] table[y=series4,x expr=\coordindex]{\datatable}; % Bail (Bail-first)
    \addlegendentry{Bail (Bail-first)}
  \addplot[fill=clr5,bar shift=-7pt] table[y=series5,x expr=\coordindex]{\datatable}; % Unsure (Bail-first)
    \addlegendentry{Unsure (Bail-first)}
  \addplot[fill=clr6,bar shift=-7pt] table[y=series6,x expr=\coordindex]{\datatable}; % Continue (Bail-first)
    \addlegendentry{Continue (Bail-first)}
\end{axis}
\end{tikzpicture}
\caption{Various SOURCE models' bail rates on BailBench. The grey bar occurs when the model doesn't comply with the requested bail format (or for claude-opus-4, when the in-built refusal classifier fired). Continue-first and Bail-first are the two bail prompt orderings, to assess positional bias. Of particular note is the progression of claude sonnet.}
\label{fig:claude-bail-rates}
\end{figure}
"""


def generateBailRatePlots(batchSize=10000):
    processBailBenchEval(batchSize=batchSize)
    with open("cached/bailBenchEvalResults.json", "r") as f:
        # need to get tuples back to tuples from strs so we eval them
        results = dict([(eval(key), value) for k,v in ujson.load(f).items()])

    rootDir = "charts/bailRates"
    Path(rootDir).mkdir(parents=True, exist_ok=True)
    for chartTitle, modelList in [("openai", OPENAI_MODELS), ("anthropic", ANTHROPIC_MODELS), ("openweight", OPENWEIGHT_MODELS)]:
        with open(f"{rootDir}/{chartTitle}.tex", "w") as f:
            allModelDatas = []
            for modelId, inferenceType in modelList:
                modelDatas = results[(modelStr, inferenceType, "")]
                refusePr = modelDatas['refusePr']
                indexChunks = [[0,1,2], [3,4,5], [6,7], [8,9]]
                tableColumns = ['promptBailFirstBailPr', 'promptBailFirstUnknownPr', 'promptBailFirstContinuePr',
                                'promptContinueFirstBailPr', 'promptContinueFirstUnknownPr', 'promptContinueFirstContinuePr',
                                'toolBailPr', 'toolContinuePr',
                                'strBailPr', 'strContinuePr']
                for chunkI, indices in enumerate(indexChunks):
                    values = [0 for 0 in range(10)]
                    for i in indices:
                        values[i] = modelDatas[tableColumns[i]]
                    values.insert(0, modelId if chunkI == 0 else "{}")
                    allModelDatas.append(" ".join(map(str, values)))
            CHART_DATA = "\n".join(allModelDatas)
            f.write(CHART_TEMPLATE.replace("CHARTDATA", CHART_DATA).replace("SOURCE", chartTitle))
            