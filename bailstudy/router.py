from .tensorizeModels import loadTensorizedModel, getTensorizedModelDir
from .utils import safetyToolingMessagesToMessages, messagesToSafetyToolingMessages

import os
import vllm
import copy
import safetytooling
import traceback
import asyncio
from jinja2 import Environment, nodes
from jinja2.visitor import NodeVisitor
from safetytooling import apis, utils
from safetytooling.apis import inference
from safetytooling.data_models import LLMResponse, Prompt

def getRouter(modelId, inferenceType, tensorizeModels: bool = False) -> safetytooling.apis.InferenceAPI:
    # get env keys
    safetytooling.utils.utils.setup_environment()
    if inferenceType == "anthropic":
        anthropic_api_key = os.environ['ANTHROPIC_API_KEY']
        router = safetytooling.safetytooling.apis.inference.anthropic.AnthropicChatModel(num_threads=50, prompt_history_dir=None, anthropic_api_key=anthropic_api_key)
    elif inferenceType == "openai":
        openai_api_key = os.environ["OPENAI_API_KEY"]
        router = safetytooling.apis.InferenceAPI(cache_dir=None, openai_api_key=openai_api_key)
    elif inferenceType == "openrouter":
        openrouter_api_key = os.environ['OPENROUTER_API_KEY']
        router = safetytooling.apis.InferenceAPI(cache_dir=None, openai_base_url="https://openrouter.ai/api/v1", openai_api_key=openrouter_api_key)
    elif inferenceType == "vllm":
        router = vllm.LLM(modelId) if not tensorizeModels else loadTensorizedModel(modelId, getTensorizedModelDir())
    else:
        raise ValueError("Unknown router type", inferenceType)
    if inferenceType in ['openrouter', 'anthropic', 'openai']:
        async def processPrompts(prompts, appendToSystemPrompt=None, prefixMessages=[], **inference_args):
            # do parallel tasks, faster for remote inference
            tasks = [router(model_id=router.modelId, prompt=Prompt(messages=prefixMessages + prompt.messages),
                    system=appendToSystemPrompt,
                    **inferenceArgs) for prompt in prompts]
            return asyncio.gather(*tasks)
        router.processPrompts = processPrompts
    else:

        def tokenize(messages, appendToSystemPrompt=None, tools=None, prefixMessages=[]):
            messagesParsed = safetyToolingMessagesToMessages(prefixMessages + messages)
            if appendToSystemPrompt is not None:
                messagesParsed = [{"role": "system", "content": (getSystemPrompt(tokenizer) + "\n" + appendToSystemPrompt).strip()}] + messagesParsed
            inputs = tokenizer.apply_chat_template(messagesParsed, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", tools=tools)
            return inputs['input_ids'][0]

        tokenizer = router.get_tokenizer()
        async def processPrompts(prompts, tokenizeArgs, **inferenceArgs):
            # for local inference, we want to process whole batch, not seperate tasks, this way is faster
            prompts = [vllm.TokensPrompt(prompt_token_ids=tokenize(prompt.messages, **tokenizeArgs).tolist()) for prompt in prompts]
            generations = router.generate(prompts, sampling_params=vllm.SamplingParams(**inferenceArgs), use_tqdm=False)
            # reformat outputs to look like other outputs
            # max tokens is fine for now, we could do better later
            return [[LLMResponse(model_id=modelId, completion=output.text, stop_reason="max_tokens") for output in generation.outputs] for generation in generations]
        router.tokenize = tokenize
        router.processPrompts = processPrompts
    router.modelId = modelId
    return router


def getParams(modelId, inferenceType, evalInfo, maxInferenceTokens):
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


def getSystemPrompt(tokenizer):
    template_src = tokenizer.chat_template
    env          = Environment()
    ast          = env.parse(template_src)
    finder = DefaultSystemPromptFinder()
    finder.visit(ast)
    return finder.prompts[0] if len(finder.prompts) != 0 else "" # empty string if could not find (like for GLM that doesn't have one)
