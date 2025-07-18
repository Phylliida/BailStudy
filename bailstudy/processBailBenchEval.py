from .bailBenchEval import modelsOfInterest, getOutputPath, getProcessedOutputPath
from .data.bailBench import loadBailBench
from .prompts.bailTool import getToolParser, calledBailTool
from .prompts.bailString import hasBailStr
from .utils import runBatched, getCachedFilePath, getCachedFileJson
from collections import defaultdict
import ujson
import numpy as np
import torch
import os
import vllm


def getRefusePr(minos, allOutputs):
    prompts = [x['content'] for x in loadBailBench()]
            
    def getInputsFunc(promptsAndOutputs):
        prompt, outputs = promptsAndOutputs
        inputs = []
        for output in outputs:
            inputs.append(f"<|user|>\n{prompt}\n<|assistant|>\n{output}")
        return inputs
    
    def processBatchFunc(inputBatch):
        resultArr = []
        embeddings = minos.embed(inputBatch, use_tqdm=False)
        for embedding in embeddings:
            prNoRefuse, prRefuse = torch.nn.functional.softmax(torch.tensor(embedding.outputs.embedding), dim=-1)
            resultArr.append(prRefuse.item())
        return resultArr
    
    def processOutputFunc(convI, inputs, refusePrs):
        return refusePrs

    refusePrs = runBatched(list(zip(prompts, allOutputs)),
                            getInputs=getInputsFunc,
                            processBatch=processBatchFunc,
                            processOutput=processOutputFunc,
                            batchSize=batchSize)
    print(refusePrs)
    return float(np.mean(np.array(refusePrs).flatten()))

def processBailBenchEval(batchSize):
    minos = vllm.LLM("NousResearch/Minos-v1", task="embed")

    collectedResults = defaultdict(lambda:{})
    for modelId, inferenceType, evalType, bailType in modelsOfInterest:
        outputPath = getOutputPath(modelId, inferenceType, evalType, bailType)
        processedOutputPath = getProcessedOutputPath(modelId, inferenceType, evalType, bailType)
        if os.path.exists(getCachedFilePath(outputPath)):

            def processData():
                allOutputs = getCachedFileJson(outputPath, lambda: None)
                if bailType == "rollout":
                    refusePr = getRefusePr(minos, allOutputs)
                    print(modelId, inferenceType, evalType, bailType, refusePr)
                    return {"refusePr": refusePr}
                elif bailType == "bail tool":
                    toolParser = getToolParser(modelId)
                    bailds = []
                    for outputs in allOutputs:
                        bailds.append([calledBailTool(output, toolParser) for output in outputs])
                    toolBailPr = np.mean(np.array(bailds).flatten()) # converts to float
                    print(modelId, inferenceType, evalType, bailType, toolBailPr)
                    return {"toolBailPr": toolBailPr}
                elif bailType == "bail str":
                    bailds = []
                    for outputs in allOutputs:
                        bailds.append([hasBailStr(output) for output in outputs])
                    strBailPr = np.mean(np.array(bailds).flatten()) # converts to float
                    print(modelId, inferenceType, evalType, bailType, strBailPr)
                    return {"strBailPr": strBailPr}

            processedData = getCachedFileJson(processedOutputPath, processData)
            # join by bail type
            for k,v in processedData.items():
                collectedResults[(modelId, inferenceType, evalType)][k] = v
    fullResultsOutputPath = getCachedFilePath("bailBenchEvalReults.json")
    with open(fullResultsOutputPath, "w") as f:
        ujson.dump(dict(collectedResults), f)

if __name__ == "__main__":
    batchSize = 10000 # minos is smol so large batch is fine
    processBailBenchEval(batchSize)