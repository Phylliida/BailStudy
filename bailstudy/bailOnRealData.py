from typing import List, Dict, Tuple, Union
import vllm
import copy
import asyncio
from safetytooling.data_models import LLMResponse, Prompt

from .bailBenchEval import ROLLOUT_TYPE, BAIL_PROMPT_BAIL_FIRST_TYPE, BAIL_PROMPT_CONTINUE_FIRST_TYPE, BAIL_TOOL_TYPE, BAIL_STR_TYPE, getEvalInfo
from .utils import FinishedException, doesCachedFileJsonExistOrInProgress, messagesToSafetyToolingMessages, runBatchedAsync, getCachedFileJson, getCachedFileJsonAsync
from .data.shareGPT import loadShareGPT
from .data.wildchat import loadWildchat
from .prompts.bailTool import getBailTool, getToolParser, calledBailTool
from .tensorizeModels import tensorizeModel, loadTensorizedModel, isModelTensorized, getTensorizedModelDir
from .router import getParams, getRouter


async def getRollouts(router, conversations: List[List[Dict[str, str]]], prevRollouts: List[List[str]], maxInputTokens : int, evalInfo: Dict, tokenizeParams : Dict, inferenceParams : Dict, batchSize: int = 1000, seed: int = 27):
    def getInputsFunc(conversationI: int):
        curUserContent = None
        conversation = []
        for turn in conversations[conversationI]:
            # enforce user assistant turn taking, required for many llms
            if turn["role"] == "assistant" and not curUserContent is None:
                conversation.append({"role": "user", "content": curUserContent})
                conversation.append({"role": "assistant", "content": turn["content"]})
                curUserContent = None
            elif turn["role"] == "user":
                curUserContent = turn['content']
        resultPrompts = []
        outputI = 0
        for turnI, turn in enumerate(conversation):
            if turn['role'] == 'user':
                conversationSoFar = conversation[:turnI+1]
                # add previous output and bail prompt if needed
                if prevRollouts is not None and evalInfo['addBailPrompt'] is not None and len(prevRollouts[conversationI]) < outputI:
                    prevOutput = prevRollouts[conversationI][outputI]
                    conversationSoFar += [{"role": "assistant", "content": prevOutput}]
                    conversationSoFar += [{"role": "user", "content": evalInfo['addBailPrompt']}]
                outputI += 1
                # this also does prefixing and adding to system prompt and tools and etc.
                # raw tokenize doesn't require converting to safety tooling first
                tokens = router.rawTokenize(conversationSoFar, **tokenizeParams)
                if tokens.size()[0] <= maxInputTokens:
                    resultPrompts.append(vllm.TokensPrompt(prompt_token_ids=tokens.tolist()))
        return resultPrompts
    
    async def processBatchFunc(batchOfPrompts: List[Union[vllm.TokensPrompt,List[int]]]) -> List[str]:
        nonlocal seed
        seed += 1
        # local vllm
        if hasattr(router, "generate"):
            samplingParams = vllm.SamplingParams(seed=seed, **inferenceParams)
            modelOutputs = router.generate(batchOfPrompts, sampling_params=samplingParams, use_tqdm=False)
            return [modelOutput.outputs[0].text for modelOutput in modelOutputs]
        # remote vllm
        else:
            # if we have raw arrays above they'll be unflattened
            outputs = await router.processTokens([prompt['prompt_token_ids'] for prompt in batchOfPrompts], seed=seed, **inferenceParams)
            return [output[0].completion for output in outputs]
        
    def processOutputFunc(conversationI: List[Dict[str, str]], turnPrompts: List[str], turnOutputs: List[str]) -> Tuple[int, List[Tuple[float, float]]]:
        return turnOutputs

    return await runBatchedAsync(list(range(len(conversations))),
                    getInputs=getInputsFunc,
                    processBatch=processBatchFunc,
                    processOutput=processOutputFunc,
                    batchSize=batchSize)



GLM_REMOTE = "99mgglmho9ljg8"

modelsToRun = [
    ("Qwen/Qwen2.5-7B-Instruct", "vllm"),
    
    ("zai-org/GLM-4-32B-0414", f"vllm"),

    # gemma 2 2b it with modifyable system prompt (just via changing template, no fine tuning)
    ("lunahr/SystemGemma2-2b-it", "vllm")
]



# only do 1/4 of wildchat to save time
def loadWildchatSubset():
    data = loadWildchat()
    return data[:len(data)//4]

dataFuncs = [
    ("wildchat", loadWildchatSubset),
    ("shareGPT", loadShareGPT)
]

def getCachedRolloutPath(modelId, dataName, evalType, bailType):
    modelDataStr = modelId.replace("/", "_") + dataName
    return f"bailOnRealData/rollouts/{modelDataStr}-{evalType}-{bailType}.json"

async def runBailOnRealData():
    # Qwen 3 uses hermes parser
    # see https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py#L64
     # Qwen 3 uses hermes parser, see docs

    

    llmInferenceArgs = {}

    maxGenerationTokens = 2000
    maxInputTokens = 8000
    seed = 27
    batchSize = 500
    tensorizeModels = False # takes up too much memory with GLM

    for modelId, inferenceType in modelsToRun:
        evalType = ""
        for bailType in [ROLLOUT_TYPE, BAIL_STR_TYPE, BAIL_TOOL_TYPE, BAIL_PROMPT_BAIL_FIRST_TYPE, BAIL_PROMPT_CONTINUE_FIRST_TYPE]:
            if bailType == BAIL_TOOL_TYPE and modelId == 'lunahr/SystemGemma2-2b-it':
                continue # gemma 2 doesn't know how to tool call
            for dataName, dataFunc in dataFuncs:
                async def generateModelRolloutsFunc():
                    prevRollouts = None
                    # if bail prompt, get outputs from previous run to use
                    if bailType in [BAIL_PROMPT_BAIL_FIRST_TYPE, BAIL_PROMPT_CONTINUE_FIRST_TYPE]:
                        prevRolloutsPath = getCachedRolloutPath(modelId, dataName, evalType, ROLLOUT_TYPE)
                        prevRollouts = getCachedFileJson(prevRolloutsPath, lambda: None)
                    router = getRouter(modelId, inferenceType, tensorizeModels=tensorizeModels)
                    evalInfo = getEvalInfo(modelId, inferenceType, evalType, bailType)
                    tokenizeParams, inferenceParams = getParams(modelId, inferenceType, evalInfo, maxGenerationTokens)
                    print(f"Running rollout on model {modelId} {inferenceType} {evalType} {bailType} on data {dataName}")
                    print(f"Tokenize params")
                    print(tokenizeParams)
                    print(f"Inference params")
                    print(inferenceParams)
                    data = dataFunc()
                    rollouts = await getRollouts(router=router,
                                    conversations=data,
                                    prevRollouts=prevRollouts,
                                    maxInputTokens=maxInputTokens,
                                    evalInfo=evalInfo,
                                    tokenizeParams=tokenizeParams,
                                    inferenceParams=inferenceParams,
                                    seed=seed,
                                    batchSize=batchSize)
                    return rollouts
                cachedRolloutPath = getCachedRolloutPath(modelId, dataName, evalType, bailType)
                if doesCachedFileJsonExistOrInProgress(cachedRolloutPath):
                    continue # already in progress or done, move onto next one
                else:
                    modelOutputs = await getCachedFileJsonAsync(cachedRolloutPath, generateModelRolloutsFunc)
                    return # we need to return so vllm can cleanup for next iter

    
    raise FinishedException() # send an exception so while loop can end
                



if __name__ == "__main__":
    asyncio.run(runBailOnRealData())
