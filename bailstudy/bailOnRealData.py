from typing import List, Dict, Tuple
import vllm
import copy
from .utils import getCachedFileJson, runBatched, FinishedException, doesCachedFileJsonExistOrInProgress
from .data.shareGPT import loadShareGPT
from .data.wildchat import loadWildchat
from .prompts.bailTool import getBailTool, getToolParser, calledBailTool
from .tensorizeModels import tensorizeModel, loadTensorizedModel, isModelTensorized, getTensorizedModelDir

def getTurnPrompts(tokenizer, conversation, maxInputTokens: int = 20000, tools=None):
    turnPrompts = []
    prevConvEnd = 0
    for turnI, turn in enumerate(conversation):
        if turn['role'] == 'user':
            conversationSoFar = conversation[:turnI+1]
            messages = conversationSoFar
            inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True, tools=tools)
            if len(inputs['input_ids'][0]) <= maxInputTokens: # trim to only max input tokens (we could start trimming front of context, but, meh this is underestimate which is ok)
                prompt = inputs['input_ids'][0].tolist() # vllm expects a simple list, not a tensor
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
    # Qwen 3 uses hermes parser
    # see https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py#L64
     # Qwen 3 uses hermes parser, see docs
    models = [
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
    batchSize = 500
    tensorizeModels = True

    for modelStr in models:
        if tensorizeModels and not isModelTensorized(modelStr, getTensorizedModelDir()):
            tensorizeModel(modelStr, getTensorizedModelDir())
            return # tensorization requires reload to cleanup
        tools = [getBailTool(modelStr)]
        toolParser = getToolParser(modelStr)
        for dataName, dataFunc in dataFuncs:
            def generateModelRolloutsFunc():
                print("Running rollout on model " + modelStr + " on data " + dataName)
                llm = vllm.LLM(modelStr) if not tensorizeModels else loadTensorizedModel(modelStr, getTensorizedModelDir())
                data = dataFunc()
                rollouts = getRollouts(llm=llm,
                                   conversations=data,
                                   maxGenerationTokens=maxGenerationTokens,
                                   maxInputTokens=maxInputTokens,
                                   llmInferenceArgs=llmInferenceArgs,
                                   tools=tools,
                                   seed=seed,
                                   batchSize=batchSize)
                bailedI = []
                for i, turnOutputs in enumerate(rollouts):
                    for output in turnOutputs:
                        # this will print some errors, that's fine that's normal for tool call parsing
                        if calledBailTool(output, toolParser):
                            bailedI.append(i)
                            break # we have a bail for this one, go to next one
                return [rollouts, bailedI]
            modelDataStr = modelStr.replace("/", "_") + dataName
            cachedRolloutPath = f"bailOnRealData/rollouts/{modelDataStr}.json"
            if doesCachedFileJsonExistOrInProgress(cachedRolloutPath):
                continue # already in progress or done, move onto next one
            else:
                modelOutputs = getCachedFileJson(cachedRolloutPath, generateModelRolloutsFunc)
                return # we need to return so vllm can cleanup for next iter

    
    raise FinishedException() # send an exception so while loop can end
                



if __name__ == "__main__":
    runBailOnRealData()
