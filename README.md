# BailStudy
Code for reproducing the experiments in our paper:

The LLM Has Left The Chat: Empirical Evidence of Bail Preferences in Large Language Models

## How to run experiments

First, install repo and dependencies

```
pip install git+https://github.com/Phylliida/BailStudy.git
```

Then make an `.env` file where you plan on running the experiments, it should look something like [https://github.com/safety-research/safety-tooling/blob/main/.env.example](https://github.com/safety-research/safety-tooling/blob/main/.env.example). In particular, it needs an Anthropic API key and an OpenAI API key.

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

## How to generate plots of data

```
python -m bailstudy.generatePlots
```

It may take a few days to complete as it processes the raw outputs into intermediate files, but only needs enough GPU to run Minos (ModernBERT-large). It caches the results as it goes along so future runs will be quick. Once it is done, your plots will be in ./plots as tex files. (I had to do lots of manual fixup of them afterwards to get them paper ready, but this will give you raw data and functioning plots etc.)

## How to visualize all the raw transcripts

There's an old version of this code designed for html viewing. AFTER YOU FINISH generatePlots, if you run

```
python -m bailstudy.oldCodePorting
```

Then you'll get output data

```
./cached/mergedbailnoswap3
./cached/mergedbailswapped3
```

This data can then be provided to the static html files, see [https://github.com/Phylliida/phylliida.github.io/tree/master/modelwelfare/bailstudy](https://github.com/Phylliida/phylliida.github.io/tree/master/modelwelfare/bailstudy) and [https://www.phylliida.dev/modelwelfare/bailstudy/](https://www.phylliida.dev/modelwelfare/bailstudy/)

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

If that does not fix the issue, try

```
pip install vllm==0.8.5 --upgrade
```

## Failure to load minos, in generatePlots

Minos needs an older version of vllm, try

```
pip install vllm==0.8.5 --upgrade
```

## ValueError: too many values to unpack (expected 2) (in flash infer)

This is a bug in flash infer, just remove flash infer
```
pip uninstall flashinfer-python
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
