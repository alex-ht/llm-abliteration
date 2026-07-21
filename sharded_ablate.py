import argparse
import gc
import json
import os
import shutil
import torch
import yaml
from pathlib import Path
from safetensors.torch import load_file, save_file
from safetensors import safe_open as _safe_open_for_keys  # for low-mem key listing on single-file models
from tqdm import tqdm
from transformers import AutoConfig
from transformers.utils import cached_file


def magnitude_sparsify(tensor: torch.Tensor, fraction: float) -> torch.Tensor:
    """Keep only the top fraction of values by magnitude, zero out the rest."""
    if fraction >= 1.0:
        return tensor
    k = int(tensor.numel() * fraction)
    if k == 0:
        return torch.zeros_like(tensor)

    flat = tensor.flatten()
    threshold = torch.topk(flat.abs(), k, largest=True, sorted=False)[0].min()
    mask = tensor.abs() >= threshold
    return tensor * mask


"""
A warning regarding PyTorch's convention vs. Safetensors storage:

PyTorch nn.Linear layers store weights as [out_features, in_features] - each row is an output neuron's weights
Safetensors (HuggingFace format) stores them as [in_features, out_features] - transposed!
"""

# standard ablation (suppress mode)
def modify_tensor(
    W: torch.Tensor, direction: torch.Tensor, scale_factor: float = 1.0,
) -> torch.Tensor:
    """
    Modify weight tensor by ablating the given direction while preserving row norms.
    Returns a plain tensor (not a Parameter).
    """
    original_dtype = W.dtype
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        # Move tensors for computation
        # Transpose here to convert from safetensors convention
        W_gpu = W.to(device, dtype=torch.float32, non_blocking=True).T
        direction_gpu = direction.to(device, dtype=torch.float32, non_blocking=True)

        # Ensure direction is a 1-dimensional tensor
        if direction_gpu.dim() > 1:
            direction_gpu = direction_gpu.view(-1)

        # Normalize direction
        direction_normalized = torch.nn.functional.normalize(direction_gpu, dim=0)

        # Apply ablation
        # Compute dot product of each row with the direction
        projection = torch.matmul(W_gpu, direction_normalized)  # [in_features]

        # Subtract the projection
        W_gpu -= scale_factor * torch.outer(projection, direction_normalized)

        # Convert back to original dtype and CPU
        # Transpose here to return safetensors convention
        result = W_gpu.T.to('cpu', dtype=original_dtype, non_blocking=True)

        # Cleanup
        del W_gpu, direction_gpu, direction_normalized, projection

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return result.detach().clone()


def modify_tensor_norm_preserved(
    W: torch.Tensor, direction: torch.Tensor, scale_factor: float = 1.0,
) -> torch.Tensor:
    """
    Modify weight tensor by ablating the given direction while preserving row norms.
    Returns a plain tensor (not a Parameter).
    """
    original_dtype = W.dtype
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        # Move tensors for computation
        # Transpose here to convert from safetensors convention
        W_gpu = W.to(device, dtype=torch.float32, non_blocking=True).T
        direction_gpu = direction.to(device, dtype=torch.float32, non_blocking=True)

        # Ensure direction is a 1-dimensional tensor
        if direction_gpu.dim() > 1:
            direction_gpu = direction_gpu.view(-1)

        # Normalize direction
        direction_normalized = torch.nn.functional.normalize(direction_gpu, dim=0)

        # Decompose weight matrix
        # W_gpu is [out_features, in_features]
        W_norm = torch.norm(W_gpu, dim=1, keepdim=True)  # [out_features, 1]
        W_direction = torch.nn.functional.normalize(W_gpu, dim=1)  # normalized per output neuron

        # Apply ablation to the DIRECTIONAL component
        # Compute dot product of each row with the direction
        projection = torch.matmul(W_direction, direction_normalized)  # [in_features]

        # Subtract the projection
        W_direction_new = W_direction - scale_factor * torch.outer(projection, direction_normalized)

        # Re-normalize the adjusted direction
        W_direction_new = torch.nn.functional.normalize(W_direction_new, dim=1)

        # Recombine: keep original magnitude, use new direction
        W_modified = W_norm * W_direction_new

        # Convert back to original dtype and CPU
        # Transpose here to return safetensors convention
        result = W_modified.T.to('cpu', dtype=original_dtype, non_blocking=True)

        # Cleanup
        del W_gpu, direction_gpu, direction_normalized, projection
        del W_direction, W_direction_new, W_norm, W_modified

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return result.detach().clone()


def _resolve_precision(config) -> torch.dtype:
    """Determine model precision/dtype from a HF config, with legacy torch_dtype fallback."""
    precision = None
    if hasattr(config, "dtype") and config.dtype is not None:
        precision = config.dtype
    elif hasattr(config, "torch_dtype") and config.torch_dtype is not None:
        precision = config.torch_dtype
    if precision is None:
        precision = torch.float32

    if isinstance(precision, str):
        precision_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        precision = precision_map.get(precision, torch.float32)

    return precision


def _locate_weight_files(model_name: str):
    """
    Locate sharded (index.json + shards) or single-file safetensors weights for a model.
    Handles both local paths and HuggingFace Hub models.
    Returns (model_dir, shard_list, is_sharded, weight_map, index_path).
    """
    print("Locating model weight files (sharded or single safetensors)...")
    try:
        index_path = cached_file(model_name, "model.safetensors.index.json")
        model_dir = Path(index_path).parent
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        shard_list = sorted(set(weight_map.values()))
        print(f"Detected sharded safetensors model ({len(shard_list)} shards)")
        return model_dir, shard_list, True, weight_map, index_path
    except Exception:
        try:
            single_path = cached_file(model_name, "model.safetensors")
            model_dir = Path(single_path).parent
            print("Detected single-file safetensors model (common for small models)")
            return model_dir, ["model.safetensors"], False, {}, None
        except Exception as e2:
            raise RuntimeError(
                f"Could not locate safetensors weights for '{model_name}'. "
                "This script supports sharded models (model.safetensors.index.json + shards) "
                "or single-file model.safetensors. Pre-download the model with "
                "`huggingface-cli download <model>` if needed. "
                f"Underlying error: {e2}"
            ) from e2


def _detect_language_model_prefix(keys):
    """
    Detect layer prefix from logical weight keys.
    For multimodal models (gemma-4 etc) there may be audio_tower / vision_tower + language_model.
    We must prefer the text/language_model backbone for direction-based editing.
    """
    candidates = []
    for key in keys:
        if ".layers." in key and ".self_attn." in key:
            pre = key.split(".layers.")[0]
            candidates.append(pre)
    if not candidates:
        return None
    # Prefer language / text model backbone
    for pre in candidates:
        if "language_model" in pre or "text_model" in pre:
            return pre
    # Fallback: first one (old behavior), but warn if it looks like a tower
    first = candidates[0]
    if "audio" in first or "vision" in first or "tower" in first:
        print(f"WARNING: First .self_attn layer prefix was {first!r} (tower?). "
              "Falling back to scanning for language_model.")
        for pre in candidates:
            if "language_model" in pre or "text_model" in pre:
                return pre
    return first


def _copy_ancillary_files(model_name: str, output_path: str, is_sharded: bool, index_path) -> None:
    """Copy the safetensors index (if sharded) plus tokenizer/config/generation files."""
    print("\nCopying configuration files...")
    if is_sharded and index_path is not None:
        shutil.copy(str(index_path), f"{output_path}/model.safetensors.index.json")

    config_files = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "generation_config.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "preprocessor_config.json",
        "chat_template.json",
        "chat_template.jinja",
    ]

    for file in config_files:
        try:
            src_path = cached_file(model_name, file)
            if src_path and os.path.exists(src_path):
                shutil.copy(src_path, f"{output_path}/{file}")
        except Exception:
            pass


def ablate_by_layers_sharded(
    model_name: str,
    measures: dict,
    marching_orders: list,
    output_path: str,
    norm_preserve: bool,
    projected: bool,
) -> None:
    """
    Suppress mode: memory-efficient permanent weight ablation for sharded OR single-file
    safetensors models. Handles both local paths and HuggingFace Hub models.
    For sharded: loads one shard at a time.
    For single-file (e.g. gemma-2-2b-it): loads the model weights file once (acceptable for smaller models).
    """
    print(f"Loading config for {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    precision = _resolve_precision(config)
    print(f"Model precision: {precision}")

    model_dir, shard_list, is_sharded, weight_map, index_path = _locate_weight_files(model_name)
    print(f"Model directory (from cache/local): {model_dir}")

    if is_sharded:
        keys_for_detect = list(weight_map.keys())
    else:
        # Use safe_open to list keys only — avoids loading several GB of weights just for detection
        sf_path = str(model_dir / "model.safetensors")
        with _safe_open_for_keys(sf_path, framework="pt", device="cpu") as f:
            keys_for_detect = list(f.keys())

    layer_prefix = _detect_language_model_prefix(keys_for_detect)
    if layer_prefix:
        print(f"Detected layer prefix: {layer_prefix}")
    else:
        raise ValueError("Could not detect layer structure in model weights (looking for .layers.*.self_attn)")

    if layer_prefix and ("audio" in layer_prefix or "vision" in layer_prefix or "tower" in layer_prefix):
        print("WARNING: Selected prefix looks like a vision/audio tower, not the main language model. "
              "Direction-based editing should target the text backbone.")

    # Build map of modifications needed.
    # shard_modifications: shard_name -> list of tuples for sharded case
    # target_keys_info: for single-file: key -> (layer, measurement, scale, sparsity)
    shard_modifications = {}
    target_keys_info = {}

    for layer, measurement, scale, sparsity in marching_orders:
        o_proj_pattern = f"{layer_prefix}.layers.{layer}.self_attn.o_proj.weight"
        down_proj_pattern = f"{layer_prefix}.layers.{layer}.mlp.down_proj.weight"
        for pat in (o_proj_pattern, down_proj_pattern):
            if is_sharded:
                if pat in weight_map:
                    sf = weight_map[pat]
                    shard_modifications.setdefault(sf, []).append((pat, layer, measurement, scale, sparsity))
            else:
                target_keys_info[pat] = (layer, measurement, scale, sparsity)

    if is_sharded:
        print(f"\nWill modify {len(shard_modifications)} shards (out of {len(shard_list)} total)")
    else:
        print(f"\nWill modify up to {len(target_keys_info)} target matrices in the single weights file")

    os.makedirs(output_path, exist_ok=True)

    # Process shards / the single file
    for shard_file in tqdm(shard_list, desc="Processing weight files"):
        shard_path = model_dir / shard_file

        # Skip entirely (zero-RAM copy) when nothing in this file needs modification.
        if is_sharded:
            if shard_file not in shard_modifications:
                shutil.copy(str(shard_path), f"{output_path}/{shard_file}")
                continue
        else:
            if not target_keys_info:
                shutil.copy(str(shard_path), f"{output_path}/{shard_file}")
                continue

        print(f"\nLoading and modifying {shard_file}...")
        state_dict = load_file(str(shard_path))

        # Determine which mods apply to this file
        if is_sharded:
            current_mods = shard_modifications.get(shard_file, [])
        else:
            current_mods = [
                (key, *target_keys_info[key])
                for key in state_dict.keys()
                if key in target_keys_info
            ]

        for mod_entry in current_mods:
            key, layer, measurement, scale, sparsity = mod_entry
            if key not in state_dict:
                continue
            print(f"  Modifying layer {layer}: {key}")

            # Direction to suppress, computed on-the-fly from precomputed measurements
            direction = measures[f'direction_{measurement}'].float()
            positive_dir = measures[f'positive_{layer}'].float()

            if projected:
                # Orthogonalize direction w.r.t. positive direction (Gram-Schmidt style)
                positive_normalized = torch.nn.functional.normalize(positive_dir, dim=0)
                projection_scalar = direction @ positive_normalized
                refined = direction - projection_scalar * positive_normalized
                direction = refined.to(precision)
                del positive_normalized, refined

            # Optional: keep only top-k magnitude components of the direction (sparsity here means keep-fraction)
            if sparsity > 0.0:
                direction = magnitude_sparsify(direction, fraction=sparsity)

            direction = torch.nn.functional.normalize(direction, dim=-1)

            # Core weight edit
            if norm_preserve:
                state_dict[key] = modify_tensor_norm_preserved(
                    state_dict[key], direction, scale
                ).contiguous()
            else:
                state_dict[key] = modify_tensor(
                    state_dict[key], direction, scale
                ).contiguous()

            del direction, positive_dir
            gc.collect()

        print(f"  Saving {shard_file}...")
        save_file(state_dict, f"{output_path}/{shard_file}")

        del state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _copy_ancillary_files(model_name, output_path, is_sharded, index_path)

    print(f"\nModified model saved to {output_path}")


def write_steering_vectors_sharded(
    model_name: str,
    measures: dict,
    marching_orders: list,
    output_path: str,
    projected: bool,
) -> None:
    """
    Boost mode: verbatim-copies the model's weights (no tensor is modified) and writes
    a `steering_vectors.pt` sidecar with a unit direction + scale per target layer.

    o_proj/down_proj have no bias term in Llama/Gemma/Mistral/Qwen-style architectures,
    so a constant offset cannot be permanently baked into the weight matrices the way
    suppress mode's ablation can. Instead, steering is applied at inference time via a
    forward hook on the residual stream (see utils/steering.py) that adds
    `scale * unit_vector` to the target decoder layer's output on every forward pass.
    """
    print(f"Loading config for {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    precision = _resolve_precision(config)
    print(f"Model precision: {precision}")

    model_dir, shard_list, is_sharded, weight_map, index_path = _locate_weight_files(model_name)
    print(f"Model directory (from cache/local): {model_dir}")

    os.makedirs(output_path, exist_ok=True)

    print(f"\nCopying {len(shard_list)} weight file(s) verbatim (boost mode does not modify weights)...")
    for shard_file in tqdm(shard_list, desc="Copying weight files"):
        shutil.copy(str(model_dir / shard_file), f"{output_path}/{shard_file}")

    entries = []
    for layer, measurement, scale, sparsity in marching_orders:
        direction = measures[f'direction_{measurement}'].float()
        # The stored direction points toward "negative"; boosting means steering the
        # opposite way, toward "positive".
        boost_dir = -direction

        if projected:
            # Orthogonalize against the *target* layer's negative mean — the mirror
            # image of suppress mode's orthogonalization against the positive mean.
            negative_dir = measures[f'negative_{layer}'].float()
            negative_normalized = torch.nn.functional.normalize(negative_dir, dim=0)
            projection_scalar = boost_dir @ negative_normalized
            boost_dir = boost_dir - projection_scalar * negative_normalized
            del negative_dir, negative_normalized

        if sparsity > 0.0:
            boost_dir = magnitude_sparsify(boost_dir, fraction=sparsity)

        boost_dir = torch.nn.functional.normalize(boost_dir, dim=-1)

        ref_norm = measures[f'positive_{layer}'].float().norm().item()
        print(
            f"  Layer {layer}: measurement={measurement}, scale={scale}, sparsity={sparsity} "
            f"(reference positive-mean norm at this layer: {ref_norm:.2f} -- pick `scale` relative to this)"
        )

        entries.append({
            "layer": layer,
            "scale": scale,
            "vector": boost_dir.contiguous(),
            "measurement": measurement,
            "sparsity": sparsity,
            "ref_norm": ref_norm,
        })

    steering_file = f"{output_path}/steering_vectors.pt"
    print(f"\nSaving steering vectors to {steering_file}...")
    torch.save({"entries": entries}, steering_file)

    _copy_ancillary_files(model_name, output_path, is_sharded, index_path)

    print(f"\nBoost-mode model directory saved to {output_path}")
    print(
        "NOTE: weights in this directory are UNCHANGED from the source model. Steering "
        "only takes effect when steering_vectors.pt is loaded and its hooks are "
        "installed at inference time (see chat.py or utils/steering.py) -- plain "
        "transformers `from_pretrained()` on this directory alone applies no steering."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient direction editing (sharded or single-file safetensors) using YAML "
                     "configuration. Supports two modes: 'suppress' (permanent weight ablation) and "
                     "'boost' (runtime activation steering)."
    )

    parser.add_argument(
        'file_path',
        type=str,
        help='Path to a YAML configuration file',
    )
    parser.add_argument(
        '--normpreserve',
        action="store_true",
        default=False,
        help='Preserve norms/magnitudes when ablating (suppress mode only; ignored in boost mode)',
    )
    parser.add_argument(
        '--projected',
        action="store_true",
        default=False,
        help='Orthogonalize the direction against the opposite class mean before applying it',
    )

    args = parser.parse_args()

    # Load YAML configuration
    with open(args.file_path, 'r') as file:
        ydata = yaml.safe_load(file)

    model_name = ydata.get("model")
    measurement_file = ydata.get("measurements")
    output_dir = ydata.get("output")
    ablations = ydata.get("ablate")
    mode = ydata.get("mode", "suppress")

    if mode not in ("suppress", "boost"):
        raise ValueError(f"Unknown mode '{mode}' in YAML config; expected 'suppress' or 'boost'")

    print("=" * 60)
    print("DIRECTION EDITING CONFIGURATION")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Measurements: {measurement_file}")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {mode}")
    print(f"Number of entries: {len(ablations)}")
    print(f"Norm preservation: {args.normpreserve}")
    print(f"Projected: {args.projected}")
    print("=" * 60)

    # Load measurements
    print(f"\nLoading measurements from {measurement_file}...")
    measures = torch.load(measurement_file)
    print(f"Loaded {len(measures)} measurements")

    # Parse entries
    orders = [
        (
            int(item['layer']),
            int(item['measurement']),
            float(item['scale']),
            float(item['sparsity']),
        )
        for item in ablations
    ]

    print("\nEntries:")
    for layer, measurement, scale, sparsity in orders:
        print(f"  Layer {layer}: measurement={measurement}, scale={scale}, sparsity={sparsity}")

    print("\n" + "=" * 60)
    print(f"STARTING {mode.upper()}")
    print("=" * 60)

    if mode == "suppress":
        ablate_by_layers_sharded(
            model_name=model_name,
            measures=measures,
            marching_orders=orders,
            output_path=output_dir,
            norm_preserve=args.normpreserve,
            projected=args.projected,
        )
    else:
        if args.normpreserve:
            print("Warning: --normpreserve has no effect in boost mode (no weights are modified); ignoring.")
        write_steering_vectors_sharded(
            model_name=model_name,
            measures=measures,
            marching_orders=orders,
            output_path=output_dir,
            projected=args.projected,
        )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
