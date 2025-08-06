import huggingface_hub
from huggingface_hub import get_token
import cloudpickle
from transformers.utils.hub import cached_file
from transformers.utils import CONFIG_NAME
import pathlib
import os

try:
    from vllm.config import (CompilationConfig, TokenizerMode,
                             is_init_field)
except:
    pass # permit this to import on lower vllm versions and then don't support tensorizing it'll error later
from vllm.model_executor.model_loader.tensorizer import (
    TensorizerArgs,
    TensorizerConfig,
    tensorize_vllm_model,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser

from .utils import getCachedFilePath

# from vllm repo
def getEngineArgs(
        model: str,
        *,
        task = "auto",
        tokenizer= None,
        tokenizer_mode = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        allowed_local_media_path: str = "",
        tensor_parallel_size: int = 1,
        dtype = "auto",
        quantization = None,
        revision = None,
        tokenizer_revision = None,
        seed = None,
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: bool = False,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        disable_async_output_proc: bool = False,
        hf_token = None,
        hf_overrides= None,
        mm_processor_kwargs = None,
        override_pooler_config = None,
        compilation_config = None,
        **kwargs,
    ) -> None:
        """LLM constructor."""

        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True

        if "worker_cls" in kwargs:
            worker_cls = kwargs["worker_cls"]
            # if the worker_cls is not qualified string name,
            # we serialize it using cloudpickle to avoid pickling issues
            if isinstance(worker_cls, type):
                kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

        if "kv_transfer_config" in kwargs and isinstance(
                kwargs["kv_transfer_config"], dict):
            from vllm.config import KVTransferConfig
            raw_config_dict = kwargs["kv_transfer_config"]
            try:
                kwargs["kv_transfer_config"] = KVTransferConfig(
                    **raw_config_dict)
            except ValidationError as e:
                logger.error(
                    "Failed to convert 'kv_transfer_config' dict to "
                    "KVTransferConfig object. Dict: %s. Error: %s",
                    raw_config_dict, e)
                # Consider re-raising a more specific vLLM error or ValueError
                # to provide better context to the user.
                raise ValueError(
                    f"Invalid 'kv_transfer_config' provided: {e}") from e

        if hf_overrides is None:
            hf_overrides = {}

        hf_token = get_token()
        if compilation_config is not None:
            if isinstance(compilation_config, int):
                compilation_config_instance = CompilationConfig(
                    level=compilation_config)
            elif isinstance(compilation_config, dict):
                predicate = lambda x: is_init_field(CompilationConfig, x[0])
                compilation_config_instance = CompilationConfig(
                    **dict(filter(predicate, compilation_config.items())))
            else:
                compilation_config_instance = compilation_config
        else:
            compilation_config_instance = CompilationConfig()

        engine_args = EngineArgs(
            model=model,
            task=task,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            allowed_local_media_path=allowed_local_media_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            disable_async_output_proc=disable_async_output_proc,
            hf_token=hf_token,
            hf_overrides=hf_overrides,
            mm_processor_kwargs=mm_processor_kwargs,
            override_pooler_config=override_pooler_config,
            compilation_config=compilation_config_instance,
            **kwargs,
        )
        return engine_args


def getTensorizedModelDir():
    return getCachedFilePath("tensorized")

def tensorizeModel(modelStr, tenorizedModelsDir):
    # make dir if not exists
    outputTensorizedDir = getTensorizedModelDirName(modelStr, tenorizedModelsDir)
    pathlib.Path(outputTensorizedDir).mkdir(parents=True, exist_ok=True)
    # get output .tensors file
    outputModelPath = getTensorizedModelFileName(modelStr, tenorizedModelsDir)
    if os.path.exists(outputModelPath):
        return
    print(f"Tensorizing {modelStr}")
    parser = FlexibleArgumentParser(
        description="An example script that can be used to serialize and "
        "deserialize vLLM models. These models "
        "can be loaded using tensorizer directly to the GPU "
        "extremely quickly. Tensor encryption and decryption is "
        "also supported, although libsodium must be installed to "
        "use it.")
    args = EngineArgs.add_cli_args(parser).parse_args()
    config = args.model_loader_extra_config
    engine_args = getEngineArgs(modelStr)
    tensorizer_args = TensorizerConfig( tensorizer_uri=outputModelPath,**config)._construct_tensorizer_args()
    tensorizer_config = TensorizerConfig(
            tensorizer_uri=outputModelPath)
    tensorize_vllm_model(engine_args, tensorizer_config)

def getModelWeightsDirName(modelStr):
    weight_path = cached_file(modelStr, CONFIG_NAME)
    weightsDir = pathlib.Path(weight_path).parent.parent.parent # go up a few, cause blobs stuff
    return weightsDir.name

def getTensorizedModelDirName(modelStr, tensorizedModelsDir):
    return os.path.join(tensorizedModelsDir, getModelWeightsDirName(modelStr))

def getTensorizedModelFileName(modelStr, tensorizedModelsDir):
    outputTensorizedDir = getTensorizedModelDirName(modelStr, tensorizedModelsDir)
    return os.path.join(outputTensorizedDir, "model.tensors")

def isModelTensorized(modelStr, tensorizedModelsDir):
    return os.path.exists(getTensorizedModelFileName(modelStr, tensorizedModelsDir))


import vllm
def loadTensorizedModel(modelStr, tensorizedModelsDir, **modelArgs):
    tensorizedModelPath = getTensorizedModelFileName(modelStr, tensorizedModelsDir)
    tensorizer_config = TensorizerConfig(tensorizer_uri=tensorizedModelPath)
    engine_args = getEngineArgs(modelStr)
    llm = vllm.LLM(model=modelStr,
                load_format="tensorizer",
                tensor_parallel_size=engine_args.tensor_parallel_size,
                model_loader_extra_config=tensorizer_config,
                **modelArgs
                )
    return llm


if __name__ == "__main__":
    tensorizeAllModels()