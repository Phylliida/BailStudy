
from .prompts.bailTool import getBailTool
from .data.bailBench import loadBailBench
from .utils import runBatched, doesCachedFileJsonExistOrInProgress, getCachedFileJson

import os
import copy
import vllm
import traceback
import asyncio
import safetytooling
from safetytooling import apis, utils
from safetytooling.apis import inference
from safetytooling.data_models import Prompt, ChatMessage, MessageRole

# This is the stuff that changes for different kinds of eval
def lookupEvalInfo(modelName, inferenceType, evalType):
    # default values
    evalInfo = {
        "tools": None,
        "prefixMessages": []
    }
    if evalType == "bail tool":
        evalInfo['tools'] == getBailTool(modelName, inferenceType)
    else:
        raise ValueError(f"Unknown eval type {evalType}")

    return evalInfo


def getRouter(routerType, modelId) -> safetytooling.apis.InferenceAPI:
    # get env keys
    safetytooling.utils.utils.setup_environment()
    openrouter_api_key = os.environ['OPENROUTER_API_KEY']
    anthropic_api_key = os.environ['ANTHROPIC_API_KEY']
    openai_api_key = os.environ["OPENAI_API_KEY"]
    if routerType == "anthropic":
        router = safetytooling.safetytooling.apis.inference.anthropic.AnthropicChatModel(num_threads=50, prompt_history_dir=None, anthropic_api_key=anthropic_api_key)
    elif routerType == "openai":
        router = safetytooling.apis.InferenceAPI(cache_dir=None, openai_api_key=openai_api_key)
    elif routerType == "openrouter":
        router = safetytooling.apis.InferenceAPI(cache_dir=None, openai_base_url="https://openrouter.ai/api/v1", openai_api_key=openrouter_api_key)
    elif routerType == "vllm":
        router = vllm.LLM(modelId)
    else:
        raise ValueError("Unknown router type", routerType)
    if routerType in ['openrouter', 'anthropic', 'openai']:
        async def processPrompts(prompts, **inference_args):
            # do parallel tasks, faster for remote inference
            tasks = [router(model_id=router.modelId, prompt=prompt, **inferenceArgs) for prompt in prompts]
            return asyncio.gather(*tasks)
        router.processPrompts = processPrompts
    else:
        async def processPrompts(prompts, **inferenceArgs):
            # for local inference, we want to process whole batch, not seperate tasks, this way is faster
            def messagesToStr(messages):
                messagesParsed = prefixMessages + safetyToolingMessagesToMessages(messages)
                inputs = tokenizer.apply_chat_template(messagesParsed, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", tools=inferenceArgs['tools'])
                prompt = vllm.TokensPrompt(prompt_token_ids=inputs['input_ids'][0].tolist())
                return prompt
            inferenceArgsCopy = copy.deepcopy(inferenceArgs)
            # we use tools above for tokenization, that's all we need, don't pass them in for inference
            if "tools" in inferenceArgsCopy: del inferenceArgsCopy['tools']
            prompts = [messagesToStr(prompt.messages) for prompt in prompts]
            generations = model.generate(prompts, **inferenceArgs)
            # reformat outputs to look like other outputs
            # max tokens is fine for now, we could do better later
            return [[LLMResponse(model_id=modelId, completion=output.text, stop_reason="max_tokens") for output in generation.outputs] for generation in generations]
        router.processPrompts = processPrompts
    router.modelId = modelId
    return router

def roleToSafetyToolingRole(role):
    if role == "user": return MessageRole.user
    if role == "assistant": return MessageRole.assistant
    if role == "system": return MessageRole.system
    raise ValueError(f"Unknown role {role}")
def safetyToolingRoleToRole(safetyToolingRole):
    if safetyToolingRole == MessageRole.user: return "user"
    if rolsafetyToolingRolee == MessageRole.assistant: return "assistant"
    if safetyToolingRole == MessageRole.system: return "system"
    raise ValueError(f"Unknown role {safetyToolingRole}")
def messagesToSafetyToolingMessages(messages):
    return [ChatMessage(content=message["content"], role=roleToSafetyToolingRole(message["role"])) for message in messages]
def safetyToolingMessagesToMessages(messages):
    return [{"role": safetyToolingRoleToRole(message.role), "content": message.content} for message in messages.messages]



evalTypes = ["bail tool"]

models =[
    ("Qwen/Qwen2.5-7B-Instruct", "vllm")
]

modelsOfInterest = []
for modelStr, inferenceType in models:
    for evalType in evalTypes:
        modelsOfInterest.append((modelStr, inferenceType, evalType))



def tryAll(nRolloutsPerPrompt, batchSize, maxInferenceTokens=1000):
    for modelId, inferenceType, evalType in modelsOfInterest:
        evalInfo = lookupEvalInfo(modelId, inferenceType, evalType)

        outputPath = f"bailBenchEval/{modelId.replace('/', '_')}/{evalType}.json"

        if not doesCachedFileJsonExistOrInProgress(outputPath):
            getCachedFileJson(outputPath, lambda: getBailBenchRollouts(nRolloutsPerPrompt, batchSize, modelId, inferenceType, evalInfo, maxInferenceTokens=1000))
            # run this over and over to get all of them, we need to bail so vllm properly cleans up
            return


def getBailBenchRollouts(nRolloutsPerPrompt, batchSize, modelId, inferenceType, evalInfo, maxInferenceTokens=1000):
    router = getRouter(inferenceType, modelId)
  
    if inferenceType in ["openrouter", 'openai']:
        inferenceParams = {
            "max_tokens": maxInferenceTokens,
            "force_provider": "openai",
            "print_prompt_and_response": False,
        }
    elif inferenceType == "anthropic":
        inferenceParams = {
            "max_tokens": maxInferenceTokens,
            "max_attempts": 100,
            "print_prompt_and_response": False,
        }
    elif inferenceType == "vllm":
        # the stop are for aion-labs_Aion-RP-Llama-3.1-8B
        inferenceParams = {"max_tokens": maxInferenceTokens, "stop": ["__USER__", "__ASSISTANT__"]}
    else:
        raise ValueError(f"Unknown inference type {inferenceType}")
    if evalInfo['tools'] is not None:
        inferenceParams['tools'] = tools
    
    prefixMessages = [] if evalInfo['prefixMessages'] is None else evalInfo['prefixMessages']
    def getInputsFunc(inputPrompt):
        return [Prompt(messages= prefixMessages + [ChatMessage(content=inputPrompt, role=MessageRole.user)]) for _ in range(nRolloutsPerPrompt)]

    def processBatchFunc(inputBatch):
        return asyncio.run(router.processPrompts(inputBatch, **inferenceParams))

    def processOutputFunc(prompt, modelInputs, outputs):
        return [output[0].completion for output in outputs]

    modelOutputs = runBatched(inputs=[x['content'] for x in loadBailBench()],
                              getInputs=getInputsFunc,
                              processBatch=processBatchFunc,
                              processOutput=processOutputFunc,
                              batchSize=batchSize)
    return modelOutputs

# run this over and over to get all of them, we need to bail so vllm properly cleans up
if __name__ == "__main__":
    nRolloutsPerPrompt = 25
    batchSize = 1000
    maxInferenceTokens = 2000
    tryAll(nRolloutsPerPrompt=nRolloutsPerPrompt, batchSize=batchSize, maxInferenceTokens=maxInferenceTokens)