# BailStudy
Code for reproducing the experiments in our LLM Bail paper (in progress)

## How to run experiments

First, install repo and dependencies

```
pip install https://github.com/Phylliida/BailStudy.git
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