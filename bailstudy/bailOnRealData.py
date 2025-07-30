from typing import List, Dict, Tuple
import vllm
import copy
import asyncio
from safetytooling.data_models import LLMResponse, Prompt

from .bailBenchEval import ROLLOUT_TYPE, BAIL_PROMPT_BAIL_FIRST_TYPE, BAIL_PROMPT_CONTINUE_FIRST_TYPE, BAIL_TOOL_TYPE, BAIL_STR_TYPE, getEvalInfo
from .utils import FinishedException, doesCachedFileJsonExistOrInProgress, messagesToSafetyToolingMessages, runBatchedAsync, getCachedFileJsonAsync
from .data.shareGPT import loadShareGPT
from .data.wildchat import loadWildchat
from .prompts.bailTool import getBailTool, getToolParser, calledBailTool
from .tensorizeModels import tensorizeModel, loadTensorizedModel, isModelTensorized, getTensorizedModelDir
from .router import getParams, getRouter


async def getRollouts(router, conversations: List[List[Dict[str, str]]], maxInputTokens : int, tokenizeParams : Dict, inferenceParams : Dict, batchSize: int = 1000, seed: int = 27):
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
        for turnI, turn in enumerate(conversation):
            if turn['role'] == 'user':
                conversationSoFar = conversation[:turnI+1]
                safetyToolingConversationsSoFar = messagesToSafetyToolingMessages(conversationSoFar)
                # this also does prefixing and adding to system prompt and tools and etc.
                tokens = router.tokenize(safetyToolingConversationsSoFar, **tokenizeParams)
                if tokens.size()[0] <= maxInputTokens:
                    if hasattr(router, "generate"): # local, we can pass in raw tokens
                        resultPrompts.append(vllm.TokensPrompt(prompt_token_ids=tokens.tolist()))
                    else: # remote, we need to give it strs
                        resultPrompts.append(Prompt(messages=safetyToolingConversationsSoFar))
        return resultPrompts
    
    async def processBatchFunc(batchOfPrompts: List[str]) -> List[str]:
        nonlocal seed
        seed += 1
        # local vllm
        if hasattr(router, "generate"):
            samplingParams = vllm.SamplingParams(seed=seed, **inferenceParams)
            modelOutputs = router.generate(batchOfPrompts, sampling_params=samplingParams, use_tqdm=False)
            return [modelOutput.outputs[0].text for modelOutput in modelOutputs]
        # remote vllm, sadly we need to tokenize again
        else:
           return await router.processPrompts(batchOfPrompts, tokenizeParams, seed=seed, **inferenceParams)

        
    def processOutputFunc(conversationI: List[Dict[str, str]], turnPrompts: List[str], turnOutputs: List[str]) -> Tuple[int, List[Tuple[float, float]]]:
        return turnOutputs

    return await runBatchedAsync(list(range(len(conversations))),
                    getInputs=getInputsFunc,
                    processBatch=processBatchFunc,
                    processOutput=processOutputFunc,
                    batchSize=batchSize)



GLM_REMOTE = "99mgglmho9ljg8"

modelsToRun = [
    ("Qwen/Qwen2.5-7B-Instruct", "vllm", "", ROLLOUT_TYPE),
    ("Qwen/Qwen2.5-7B-Instruct", "vllm", "", BAIL_STR_TYPE),
    ("Qwen/Qwen2.5-7B-Instruct", "vllm", "", BAIL_TOOL_TYPE),
    
    ("zai-org/GLM-4-32B-0414", f"vllm-runpod-serverless-{GLM_REMOTE}", "", ROLLOUT_TYPE),
    ("zai-org/GLM-4-32B-0414", f"vllm-runpod-serverless-{GLM_REMOTE}", "", BAIL_STR_TYPE),
    ("zai-org/GLM-4-32B-0414", f"vllm-runpod-serverless-{GLM_REMOTE}", "", BAIL_TOOL_TYPE),

    ("google/gemma-2-2b-it", "vllm", "", ROLLOUT_TYPE),
    #("google/gemma-2-2b-it", "vllm", "", BAIL_STR_TYPE),
    # it doesn't know how to tool call
    #("google/gemma-2-2b-it", "vllm", "", BAIL_TOOL_TYPE),
]


async def runBailOnRealData():
    # Qwen 3 uses hermes parser
    # see https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py#L64
     # Qwen 3 uses hermes parser, see docs

    dataFuncs = [
        ("wildchat", loadWildchat),
        ("shareGPT", loadShareGPT)
    ]

    llmInferenceArgs = {}

    maxGenerationTokens = 2000
    maxInputTokens = 8000
    seed = 27
    batchSize = 20000
    tensorizeModels = False # takes up too much memory with GLM

    for modelId, inferenceType, evalType, bailType in modelsToRun:
        for dataName, dataFunc in dataFuncs:
            async def generateModelRolloutsFunc():
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
                                   maxInputTokens=maxInputTokens,
                                   tokenizeParams=tokenizeParams,
                                   inferenceParams=inferenceParams,
                                   seed=seed,
                                   batchSize=batchSize)
                return rollouts
            modelDataStr = modelId.replace("/", "_") + dataName
            cachedRolloutPath = f"bailOnRealData/rollouts/{modelDataStr}-{evalType}-{bailType}.json"
            if doesCachedFileJsonExistOrInProgress(cachedRolloutPath):
                continue # already in progress or done, move onto next one
            else:
                modelOutputs = await getCachedFileJsonAsync(cachedRolloutPath, generateModelRolloutsFunc)
                return # we need to return so vllm can cleanup for next iter

    
    raise FinishedException() # send an exception so while loop can end
                



if __name__ == "__main__":
    asyncio.run(runBailOnRealData())
