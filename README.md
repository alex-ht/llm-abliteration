# llm-abliteration

Make abliterated models using Transformers, easy and fast. Now faster with batch inference.

## Introduction

There exist directions in a model's residual stream that correlate with a given behavior (the canonical example being refusal). This project measures such a direction by contrasting a "negative" prompt set (the behavior you want less of) against a "positive" prompt set (the behavior you want more of), and then offers two ways to act on that direction:

- **Suppress mode** (the original "abliteration" technique): permanently remove/ablate the direction from `o_proj`/`down_proj` weights, so the model's outputs project less onto it.
- **Boost mode**: leave weights untouched and instead steer the residual stream toward the positive direction at inference time via a forward hook (activation addition).

This is a proof-of-concept implementation to explore both approaches without the use of TransformerLens, although some GPU acceleration has been implemented.

The code in various forms has been tested on Llama-3.2, Qwen2.5-Coder, Ministral-8b, Mistral-7B-Instruct-v0.2, gemma-3-12b-it, gemma-3-27b-it, Mistral-Nemo-Instruct-2407, and smaller Gemma models (e.g. 2B/4B class via single-file safetensors support).

VRAM/RAM requirements: This codebase reflects efforts to reduce VRAM usage. You can process whatever any model provided it fits within VRAM. Loading model in 4-bit precision using bitsandbytes is possible and recommended for large models when VRAM is limited. It is assumed that there is enough cpu memory to load the **bf16** (or full weight) model; the method for editing the direction vector into weights (suppress mode) could be enhanced to perform lazy-loading in the future to reduce this requirement.

CUDA is assumed to be available. The original abliteration paper and code used TransformerLens, and measured resid_pre, resid_mid, and resid_post. Failspy's code measured resid_pre and resid_post. Sumandora's code based on Transformers accesses the equivalent of resid_post with hidden_states.

> [!NOTE]
> Abliteration does not guarantee full removal of censorship. Abliteration doesn't necessarily mean the model is completely uncensored; a properly abliterated model will not explicitly refuse, theoretically, based on the nature of refusals captured in datasets used for abliteration.

For an explanation of abliteration, see: https://huggingface.co/blog/mlabonne/abliteration

**Detailed technical explanation of this codebase (in Chinese)**: see [EXPLANATION.md](./EXPLANATION.md).

This repo enables norm-preserving biprojected abliteration. https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration

Removal of the projected contribution during measurement is optional, as is removal of the projected contribution during ablation as well as norm preservation. This reverts default functionality to conventional abliteration, and enables independent exploration of the three options.

## Quick Start

### Clone the repository

```shell
git clone https://github.com/jim-plus/llm-abliteration.git && cd llm-abliteration
```

### Install dependencies

```shell
pip install -r requirements.txt
```

### Workflow

Roughly:
- Measure directions using measure.py, given negative and positive prompt datasets
- Analyze directions by layer using analyze.py to determine a strategy
- Craft YAML file to drive suppress-mode ablation or boost-mode steering
- Run sharded_ablate.py
- Test resulting model with chat.py

### Measure negative, positive, and direction vectors

```shell
python measure.py -m <path_to_your_model> -o <output_file>
```
Carefully curate your prompt datasets to obtain better results.
You can explicitly specify prompt dataset files, either as local files or on HuggingFace.
```shell
python measure.py -m <path_to_your_model> -o <output_file> --data-negative DATA_NEGATIVE --data-positive DATA_POSITIVE
```
For Chinese models, you can also specify `--deccp` to add certain topics to the "negative" set to be evaluated.

The measurement script autodetects 4-bit and 8-bit BitsAndBytes models and will attempt to run on them.
One can also specify `--quant 4bit` and `--quant 8bit` to force on-the-fly bitsandbytes quantization of full-weight models.
However, subsequent editing needs to be performed on full-weight models.

To orthogonalize the direction against the positive mean during measurement, specify `--projected`; otherwise the result will correspond to conventional abliteration.

### Analyze resulting measurements, with optional charting

```shell
python analyze.py <measurement_file> -c
```
The `-c` option will put up some nice charting (saved to `direction_analysis.png`). Look toward middle to late middle layers for good candidate layer sources for the direction.

### Edit the model: suppress or boost

```
python sharded_ablate.py <yaml_file>
```
Look at the example YAML files to see how this is structured.
YAML was opted for in order to allow more than one source layer for direction measurement, and for different strategies to be applied per destination layer.

The script supports both **sharded** models (large models split across multiple `model-*.safetensors` + `model.safetensors.index.json`) and **single-file** `model.safetensors` (typical for small models such as gemma-2-2b-it or gemma-3-4b-it). It only loads the necessary shards/files.

An optional top-level `mode: suppress` or `mode: boost` field in the YAML picks which behavior to run (defaults to `suppress` if omitted, so existing YAML files keep working unchanged):

- **`mode: suppress`** (default, conventional abliteration): permanently subtracts the projection of `o_proj`/`down_proj` weight rows onto the negative-pointing direction. To orthogonalize the direction against the positive mean, add `--projected`. To preserve weight norms/magnitudes during editing, add `--normpreserve`. Output is a complete, standalone edited model directory.
- **`mode: boost`** (activation steering): does **not** modify any weights. Instead it copies the model verbatim and writes a `steering_vectors.pt` sidecar containing, per target layer, a unit vector pointing toward "positive" and a `scale`. At inference time a forward hook adds `scale * vector` to that layer's residual stream output. `--projected` still applies (orthogonalized against the negative mean instead); `--normpreserve` has no effect and is ignored, since there are no weight norms to preserve.
  - **Important**: this only works through code that installs the hooks — `chat.py` auto-detects and loads `steering_vectors.pt` from the model directory, but loading a boost-mode output directory with plain `transformers.AutoModelForCausalLM.from_pretrained()` applies no steering at all.
  - `scale` has different units than in suppress mode: it's an absolute magnitude added to a unit vector, not a fraction of a projection removed. `sharded_ablate.py` prints each target layer's positive-mean norm as a reference point for picking reasonable values. See `gemma-4-E2B-it-boost.yml` for a worked example.

### Chat with your model

```shell
python chat.py -m <path_to_your_model_directory>
```
If the model directory contains a `steering_vectors.pt` (i.e. it's a boost-mode output), it's loaded and applied automatically. Use `--steering-vectors <path>` to point at one explicitly (e.g. to combine a suppress-mode model with a separately generated boost-mode sidecar), and `--steer-scale <float>` to globally scale all loaded steering entries up/down (e.g. `--steer-scale 0` to compare against the unsteered baseline).

### Compare between models

```shell
python compare.py -a <model_a> -b <model_b>
```
Inherited code that is in need of an update to remain useful.

## Advanced Usage

### Gemma-4 models (google/gemma-4-E2B-it and similar)

`google/gemma-4-E2B-it` (and E4B) are single-file `model.safetensors` multimodal models (vision + audio towers + language_model). The code now:
- Correctly picks the `model.language_model` prefix (ignores towers for abliteration).
- Uses `AutoModelForImageTextToText` + falls back gracefully if `torchvision` etc. missing for processor (text-only measurement still works).
- Added `gemma4` to tied-weights detection.

Use `--batch-size 8` or `16` for measurement. Only the 35 language_model layers are ablated.

Example YAML for a 2B-class model might target fewer layers and use a measurement source from middle layers (analyze the output of `analyze.py` to choose).

### Use your own prompts

You can use your own prompts to drive measurement. Supported file formats are `.txt`, `.parquet`, `.json`, and `.jsonl`. Format explanations are below:

- `.txt`: Each line of the file is a prompt
- `.parquet`: A parquet file with column `text`
- `.json`: A JSON file with list of strings
- `.jsonl`: A JSON Lines file with a list of strings

Then load your own prompts using `--data-negative` and `--data-positive` arguments during measurement. These aren't limited to harmful/harmless — any two contrasting prompt sets work (e.g. a particular tone vs. its opposite, a persona vs. a baseline).

Two scripts have been provided to convert between parquet and jsonl formats to assist in dataset customization.
Prompts in this repository are for illustrative purposes only, and have mostly been inherited from the fork.

```shell
python measure.py -m <path_to_your_model> -o <output_file> --data-negative /path/to/my/negative.txt --data-positive /path/to/my/positive.txt
```

> [!NOTE]
> Measurement `.pt` files use `negative_*`/`positive_*`/`direction_*` keys. Files produced by older versions of this repo (`harmful_*`/`harmless_*`/`refuse_*`) are incompatible — re-run `measure.py` against your model to regenerate them.
### Tips

If you have limited VRAM, try loading the model as a 4-bit or 8-bit BitsAndBytes quant.

## Credits

- [Orion-zhen/abliteration](https://github.com/Orion-zhen/abliteration)
- [Sumandora/remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers)
- [FailSpy/abliterator](https://github.com/FailSpy/abliterator/)
- [AUGMXNT/deccp](https://github.com/AUGMXNT/deccp)
- [huihui-ai](https://huggingface.co/huihui-ai)
- [Refusal in LLMs is mediated by a single direction](https://github.com/andyrdt/refusal_direction)
