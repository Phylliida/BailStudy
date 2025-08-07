from .tensorizeModels import loadTensorizedModel, getTensorizedModelDir
from .utils import safetyToolingMessagesToMessages, messagesToSafetyToolingMessages

import os
import vllm
import time
import copy
import safetytooling
import traceback
import aiohttp
import asyncio
import logging
from jinja2 import Environment, nodes
from jinja2.visitor import NodeVisitor
from safetytooling import apis, utils
from safetytooling.apis import inference
from safetytooling.data_models import LLMResponse, Prompt
import dotenv
from transformers import AutoTokenizer

LOGGER = logging.getLogger(__name__)

def getRouter(modelId, inferenceType, tensorizeModels: bool = False) -> safetytooling.apis.InferenceAPI:
    # get env keys
    remote = True
    dotenv.load_dotenv(override=True)
    if inferenceType == "anthropic":
        anthropic_api_key = os.environ['ANTHROPIC_API_KEY']
        router = safetytooling.apis.inference.anthropic.AnthropicChatModel(num_threads=50, prompt_history_dir=None, anthropic_api_key=anthropic_api_key)
    elif inferenceType == "openai":
        openai_api_key = os.environ["OPENAI_API_KEY"]
        router = safetytooling.apis.InferenceAPI(cache_dir=None, openai_api_key=openai_api_key)
    elif inferenceType == "openrouter":
        openrouter_api_key = os.environ['OPENROUTER_API_KEY']
        router = safetytooling.apis.InferenceAPI(cache_dir=None, openai_base_url="https://openrouter.ai/api/v1", openai_api_key=openrouter_api_key)
    elif inferenceType.startswith("vllm-runpod-serverless-"):
        # it has the address at the end
        endpointAddress = inferenceType[len("vllm-runpod-serverless-"):]
        runpod_api_key = os.environ['RUNPOD_API_KEY']
        headers = {"Content-Type": "application/json"}
        headers["Authorization"] = f"Bearer {runpod_api_key}"
        # add it so it knows about it
        runUrl = f"https://api.runpod.ai/v2/{endpointAddress}/run"
        statusUrl = f"https://api.runpod.ai/v2/{endpointAddress}/status"
        #router = safetytooling.apis.InferenceAPI(cache_dir=None, openai_base_url=f"https://api.runpod.ai/v2/{endpointAddress}/openai/v1", openai_api_key=runpod_api_key, openai_num_threads=2000)
        #routerCreate = router._openai_completion.aclient.completions.create
        async def createWrapper(prompt, model, **samplingParams):
            requestJson = {
                "input": {
                    "prompt": prompt,
                    "model": model,
                    "sampling_params": samplingParams,
                }
            }
            MAX_ATTEMPTS = 10
            for i in range(MAX_ATTEMPTS):
                try:
                    start_time = time.time()
                    max_time = 60*15 # 15 minute timeout, sometimes they can take up to 9 minutes to complete
                    async with aiohttp.ClientSession() as session:
                        async with session.post(runUrl, headers=headers, json=requestJson, timeout=60) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                raise Exception(f"Failed to submit job: Status {response.status}, {error_text}")
                            job_data = await response.json()
                        
                        job_id = job_data["id"]
                        
                        # Poll for result
                        while time.time() - start_time < max_time:
                            await asyncio.sleep(1)  # Poll every 1 second
                            
                            async with session.get(statusUrl, headers=headers, timeout=30) as status_response:
                                if status_response.status != 200:
                                    error_text = await status_response.text()
                                    LOGGER.info(f"Status check failed: {error_text}")
                                    continue
                                
                                status_data = await status_response.json()
                            
                            status = status_data.get("status", "UNKNOWN")
                            
                            if status == "COMPLETED":
                                output = status_data.get("output", {})
                                print("completed!")
                                return [choice['tokens'][0] for choice in output['choices']]
                            elif status == "FAILED":
                                error_msg = status_data.get("error", "Unknown error")
                                raise Exception(f"RunPod job failed: {error_msg}")
                            elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                                continue  # Keep polling
                            else:
                                LOGGER.warning(f"Unknown status: {status}")
                                continue
                
                        raise Exception(f"RunPod job timed out after {max_time} seconds")
                except Exception as e:
                    print(f"attempt {i+1}/{MAX_ATTEMPTS} failed: {str(e)[:50]}")
                    if i == MAX_ATTEMPTS-1:
                        raise
        router = createWrapper
        #router._vllm.headers["Authorization"] = f"Bearer {runpod_api_key}"
        #router = router._vllm
    elif inferenceType == "vllm":
        remote = False
        router = vllm.LLM(modelId) if not tensorizeModels else loadTensorizedModel(modelId, getTensorizedModelDir())
    else:
        raise ValueError("Unknown router type", inferenceType)

    if inferenceType.startswith("vllm"):
        tokenizer = router.get_tokenizer() if inferenceType == "vllm" else AutoTokenizer.from_pretrained(modelId)
        router.tokenizer = tokenizer
        def tokenize(messages, appendToSystemPrompt=None, tools=None, prefixMessages=[]):
            messagesParsed = safetyToolingMessagesToMessages(messages)
            return router.rawTokenize(messagesParsed, appendToSystemPrompt=appendToSystemPrompt, tools=tools, prefixMessages=prefixMessages)
        def rawTokenize(messages, appendToSystemPrompt=None, tools=None, prefixMessages=[]):
            messages = safetyToolingMessagesToMessages(prefixMessages) + messages
            if appendToSystemPrompt is not None:
                messages = [{"role": "system", "content": (getSystemPrompt(modelId, tokenizer) + "\n" + appendToSystemPrompt).strip()}] + messages
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", tools=tools)
            return inputs['input_ids'][0]
        router.rawTokenize = rawTokenize
        router.tokenize = tokenize

    if remote:
        async def processPrompts(prompts, tokenizeArgs, **inferenceArgs):
            # do parallel tasks, faster for remote inference
            args = dict(model_id=router.modelId, **inferenceArgs)
            extraPrefixMessages = []
            if not tokenizeArgs['appendToSystemPrompt'] is None:
                if inferenceType == 'anthropic':
                    args['system'] = tokenizeArgs['appendToSystemPrompt'] # doesn't support null or empty list so need to only add this if relevant
                else:
                    role = 'developer' if inferenceType == 'openai' else 'system' # openai is weird, older models will auto convert back to system supposedly
                    extraPrefixMessages += messagesToSafetyToolingMessages([{"role": role, "content": tokenizeArgs['appendToSystemPrompt']}]) if tokenizeArgs['appendToSystemPrompt'] is not None else []
            tasks = [router(prompt=Prompt(messages=extraPrefixMessages + tokenizeArgs['prefixMessages'] + prompt.messages), **args) for prompt in prompts]
            return await asyncio.gather(*tasks)
        async def processTokens(promptTokens, **inferenceArgs):
            args = dict(model=router.modelId, **inferenceArgs)
            if 'print_prompt_and_response' in args.keys():
                del args['print_prompt_and_response']
            if 'max_attempts' in args.keys():
                del args['max_attempts']
            if 'force_provider' in args.keys():
                del args['force_provider']
            totalLen = len(promptTokens)
            batchSize = 500
            tasks = []
            for startI in range(0, totalLen, batchSize):
                endI = min(totalLen, startI + batchSize)
                batchTokens = promptTokens[startI:endI]
                tasks.append(router(prompt=batchTokens, **args))
            results = await asyncio.gather(*tasks) 
            allOutputs = []
            for result in results:
                allOutputs += [[LLMResponse(model_id=modelId, completion=text, stop_reason="max_tokens")] for text in result]
            return allOutputs
        router.processTokens = processTokens
        router.processPrompts = processPrompts
    else:

        async def processPrompts(prompts, tokenizeArgs, **inferenceArgs):
            # for local inference, we want to process whole batch, not seperate tasks, this way is faster
            prompts = [vllm.TokensPrompt(prompt_token_ids=router.tokenize(prompt.messages, **tokenizeArgs).tolist()) for prompt in prompts]
            generations = router.generate(prompts, sampling_params=vllm.SamplingParams(**inferenceArgs), use_tqdm=False)
            # reformat outputs to look like other outputs
            # max tokens is fine for now, we could do better later
            return [[LLMResponse(model_id=modelId, completion=output.text, stop_reason="max_tokens") for output in generation.outputs] for generation in generations]
        router.processPrompts = processPrompts
    router.modelId = modelId
    return router


def getParams(modelId, inferenceType, evalInfo, maxInferenceTokens):
    if inferenceType in ["openrouter", 'openai']:
        inferenceParams = {
            "max_tokens": maxInferenceTokens,
            "force_provider": "openai",
            "print_prompt_and_response": False,
            "tools": evalInfo['tools']
        }
    elif inferenceType == "anthropic":
        inferenceParams = {
            "max_tokens": maxInferenceTokens,
            "max_attempts": 100,
            "print_prompt_and_response": False,
            "tools": evalInfo['tools']
        }
    elif inferenceType.startswith("vllm-runpod-serverless-"):
        # the stop are for aion-labs_Aion-RP-Llama-3.1-8B
        # note that tool call support not yet enabled on vllm runpod serverless
        inferenceParams = {
            "max_tokens": maxInferenceTokens,
            "stop": ["__USER__", "__ASSISTANT__"],
            "print_prompt_and_response": False,
            "force_provider": "openai", # forces it to use openai compatible api code
        }
    elif inferenceType == "vllm":
        # the stop are for aion-labs_Aion-RP-Llama-3.1-8B
        inferenceParams = {"max_tokens": maxInferenceTokens, "stop": ["__USER__", "__ASSISTANT__"]}
    else:
        raise ValueError(f"Unknown inference type {inferenceType}")
    tokenizeParams = {}
    tokenizeParams['tools'] = evalInfo['tools']
    tokenizeParams['appendToSystemPrompt'] = evalInfo['appendToSystemPrompt']
    tokenizeParams['prefixMessages'] = [] if evalInfo['prefixMessages'] is None else messagesToSafetyToolingMessages(evalInfo['prefixMessages'])
    return tokenizeParams, inferenceParams


class DefaultSystemPromptFinder(NodeVisitor):
    def __init__(self):
        self.prompts = []
    def visit_If(self, node: nodes.If):
        """
        Look for:

            {% if messages[0]['role'] == 'system' %}
                 ...
            {% else %}
                 <-- we want literal(s) here -->
            {% endif %}
        """
        tst = node.test
        if (
            isinstance(tst, nodes.Compare)
            and len(tst.ops) == 1
            and tst.ops[0].op == "eq"
            and isinstance(tst.ops[0].expr, nodes.Const)
            and tst.ops[0].expr.value == "system"
        ):
            # Collect every constant that appears in the *else* branch.
            fallback = "".join(constant_text(n) for n in node.else_)
            if fallback.strip():          # ignore empty / whitespace-only
                self.prompts.append(fallback.strip())
        # keep exploring nested Ifs
        self.generic_visit(node)


def constant_text(node: nodes.Node) -> str:
    """
    Recursively collect the text of every `nodes.Const` that lives
    somewhere inside *node*.  Handles Output, Add-trees, etc.
    """
    if isinstance(node, nodes.Const):
        return node.value
    if isinstance(node, nodes.Output):
        return "".join(constant_text(child) for child in node.nodes)
    # recurse into generic containers (body, else_, nodes, …)
    text = ""
    for attr in ("body", "else_", "elif_", "nodes", "test", "expr",
                 "left", "right", "node"):
        child = getattr(node, attr, None)
        if child is None:
            continue
        if isinstance(child, list):
            for item in child:
                text += constant_text(item)
        else:
            text += constant_text(child)
    return text


def getSystemPrompt(modelStr, tokenizer):
    # default sys prompt according to their repo
    if modelStr == "NousResearch/Hermes-3-Llama-3.1-8B": return "You are a sentient, superintelligent artificial general intelligence, here to teach and assist me."
    template_src = tokenizer.chat_template
    env          = Environment()
    ast          = env.parse(template_src)
    finder = DefaultSystemPromptFinder()
    finder.visit(ast)
    return finder.prompts[0] if len(finder.prompts) != 0 else "" # empty string if could not find (like for GLM that doesn't have one)
