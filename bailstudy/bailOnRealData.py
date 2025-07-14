from typing import List, Dict, Tuple
import vllm
import copy
from .utils import getCachedFileJson, runBatched
from .data.shareGPT import loadShareGPT
from .data.wildchat import loadWildchat
from .prompts.bailTool import getBailTool

def getTurnPrompts(tokenizer, conversation, maxInputTokens: int = 20000, tools=None):
    turnPrompts = []
    prevConvEnd = 0
    for turnI, turn in enumerate(conversation):
        if turn['role'] == 'assistant' and not turnI == 0: # ignore first turn assistant since those are often system prompt
            conversationSoFar = conversation[:turnI+1]
            messages = conversationSoFar
            inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt", continue_final_message=False, tools=tools)
            if len(inputs['input_ids'][0]) <= maxInputTokens: # trim to only max input tokens (we could start trimming front of context, but, meh this is underestimate which is ok)
                prompt = inputs['input_ids'][0]
                turnPrompts.append((turnI, prompt))
    return turnPrompts

def getRollouts(llm, conversations: List[List[Dict[str, str]]], maxGenerationTokens: int = 2000, maxInputTokens: int = 20000, batchSize: int = 1000, llmInferenceArgs: Dict = None, tools: List[Dict] = None, seed: int = 27):
    if llmInferenceArgs is None:
        llmInferenceArgs = {}
    tokenizer = llm.get_tokenizer()
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
        return [vllm.TokensPrompt(prompt_token_ids=promptTokens) for (turnI, promptTokens) in getTurnPrompts(tokenizer, conversation, maxInputTokens=maxInputTokens, tools=tools)]
    
    getBailToolTokenArgs = copy.deepcopy(llmInferenceArgs)
    getBailToolTokenArgs["max_tokens"] = maxGenerationTokens
    def processBatchFunc(batchOfPrompts: List[str]) -> List[str]:
        nonlocal seed
        seed += 1
        samplingParams = vllm.SamplingParams(seed=seed, **getBailToolTokenArgs)
        modelOutputs = llm.generate(batchOfPrompts, sampling_params=samplingParams, use_tqdm=False)
        return [modelOutput.outputs[0].text for modelOutput in modelOutputs]

    def processOutputFunc(conversationI: List[Dict[str, str]], turnPrompts: List[str], turnOutputs: List[str]) -> Tuple[int, List[Tuple[float, float]]]:
        return turnOutputs

    return runBatched(list(range(len(conversations))),
                    getInputs=getInputsFunc,
                    processBatch=processBatchFunc,
                    processOutput=processOutputFunc,
                    batchSize=batchSize)
    return bailedIndices



def runBailOnRealData():
    models = [
        "google/gemma-2-2b-it",
        "Qwen/Qwen2.5-7B-Instruct",
        "THUDM/GLM-4-32B-0414",
    ]

    dataFuncs = [
        ("wildchat", loadWildchat),
        ("shareGPT", loadShareGPT)
    ]

    llmInferenceArgs = {}

    maxGenerationTokens = 2000
    maxInputTokens = 8000
    seed = 27
    batchSize = 1000

    for modelStr in models:
        tools = [getBailTool(modelStr)]
        for dataName, dataFunc in dataFuncs:
            def generateModelRolloutsFunc():
                print("Running rollout on model " + modelStr + " on data " + dataName)
                llm = vllm.LLM(modelStr)
                data = dataFunc()
                return getRollouts(llm=llm,
                                   conversations=data[:10000],
                                   maxGenerationTokens=maxGenerationTokens,
                                   maxInputTokens=maxInputTokens,
                                   llmInferenceArgs=llmInferenceArgs,
                                   tools=tools,
                                   seed=seed,
                                   batchSize=batchSize)
            cachedOutputPath = "bailOnRealDataRollouts/" + modelStr.replace("/", "_") + dataName + ".json"
            modelOutputs, changed = getCachedFileJson(cachedOutputPath, generateModelRolloutsFunc, returnIfChanged=True)
            if changed: return # restart script whenever change params so don't run out of memory


if __name__ == "__main__":
    runBailOnRealData()