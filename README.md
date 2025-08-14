# BailStudy
Code for reproducing the experiments in our LLM Bail paper (in progress)

## How to run experiments

First, install repo and dependencies

```
pip install git+https://github.com/Phylliida/BailStudy.git
```

Much of the code below is wrapped in while loops, this is because we need to switch vllm models.
There's not an easy way to do that (afaik) without killing python instance and restarting, so a while loop does that.

### Run "Models bail on real world data" experiments

This will take a week or two to complete, and requires lots of VRAM.
You can speed it up by running it on multiple machines (it'll automatically distribute work between them),
just make sure both machines run the code in the same directory so they can coordinate.

It will also download WildChat and ShareGPT to `./cached` (and also put model outputs there)

```
while python -m bailstudy.bailOnRealData; do :; done
```

### Run BailBench related experiments

This will gather data for all plots related to bail bench. It may take a few weeks to complete.
You can speed it up by running it on multiple machines (it'll automatically distribute work between them),
just make sure both machines run the code in the same directory so they can coordinate.

`./cached` will contain tensorized models (for quick loading during these measurements) as well as outputs from the experiments.

```
while python -m bailstudy.bailBenchEval; do :; done
```

## Troubleshooting

## generated_content does not exist

Make sure you've installed the branch of safety tooling with tool support

```
pip install git+https://github.com/safety-research/safety-tooling.git@abhay/tools
```


## Exception: Invalid prefix encountered (when doing real world data with gemma2-2b-it)

## ImportError: cannot import name 'TokenizerMode' from 'vllm.config' (/root/.venv/lib/python3.11/site-packages/vllm/config.py)

## ImportError: cannot import name 'ResponsePrompt' from 'openai.types.responses'

For these three errors, just do

```
pip install vllm --upgrade
```

## Failure to load minos, in generatePlots

Minos needs an older version of vllm, try

```
pip install vllm==0.8.5 --upgrade
```


## Model stuck on downloading

Run

```
ps -aux | grep python
```

And send `kill -9 processId` to each listed process.

Then run

```
export HF_HUB_DISABLE_XET="1"
```

and retry. See [this thread](https://github.com/huggingface/hf_transfer/issues/30#issuecomment-2878604131)