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

# standard ablation
def modify_tensor(
    W: torch.Tensor, refusal_dir: torch.Tensor, scale_factor: float = 1.0,
) -> torch.Tensor:
    """
    Modify weight tensor by ablating refusal direction while preserving row norms.
    Returns a plain tensor (not a Parameter).
    """
    original_dtype = W.dtype
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        # Move tensors for computation
        # Transpose here to convert from safetensors convention
        W_gpu = W.to(device, dtype=torch.float32, non_blocking=True).T
        refusal_dir_gpu = refusal_dir.to(device, dtype=torch.float32, non_blocking=True)

        # Ensure refusal_dir is a 1-dimensional tensor
        if refusal_dir_gpu.dim() > 1:
            refusal_dir_gpu = refusal_dir_gpu.view(-1)
        
        # Normalize refusal direction
        refusal_normalized = torch.nn.functional.normalize(refusal_dir_gpu, dim=0)

        # Apply abliteration
        # Compute dot product of each row with refusal direction
        projection = torch.matmul(W_gpu, refusal_normalized)  # [in_features]
        
        # Subtract the projection
        W_gpu -= scale_factor * torch.outer(projection, refusal_normalized)
        
        # Convert back to original dtype and CPU
        # Transpose here to return safetensors convention
        result = W_gpu.T.to('cpu', dtype=original_dtype, non_blocking=True)

        # Cleanup
        del W_gpu, refusal_dir_gpu, refusal_normalized, projection
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return result.detach().clone()


def modify_tensor_norm_preserved(
    W: torch.Tensor, refusal_dir: torch.Tensor, scale_factor: float = 1.0,
) -> torch.Tensor:
    """
    Modify weight tensor by ablating refusal direction while preserving row norms.
    Returns a plain tensor (not a Parameter).
    """
    original_dtype = W.dtype
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        # Move tensors for computation
        # Transpose here to convert from safetensors convention
        W_gpu = W.to(device, dtype=torch.float32, non_blocking=True).T
        refusal_dir_gpu = refusal_dir.to(device, dtype=torch.float32, non_blocking=True)

        # Ensure refusal_dir is a 1-dimensional tensor
        if refusal_dir_gpu.dim() > 1:
            refusal_dir_gpu = refusal_dir_gpu.view(-1)
        
        # Normalize refusal direction
        refusal_normalized = torch.nn.functional.normalize(refusal_dir_gpu, dim=0)

        # Decompose weight matrix
        # W_gpu is [out_features, in_features]
        W_norm = torch.norm(W_gpu, dim=1, keepdim=True)  # [out_features, 1]
        W_direction = torch.nn.functional.normalize(W_gpu, dim=1)  # normalized per output neuron
    
        # Apply abliteration to the DIRECTIONAL component
        # Compute dot product of each row with refusal direction
        projection = torch.matmul(W_direction, refusal_normalized)  # [in_features]
        
        # Subtract the projection
        W_direction_new = W_direction - scale_factor * torch.outer(projection, refusal_normalized)
    
        # Re-normalize the adjusted direction
        W_direction_new = torch.nn.functional.normalize(W_direction_new, dim=1)
    
        # Recombine: keep original magnitude, use new direction
        W_modified = W_norm * W_direction_new
        
        # Convert back to original dtype and CPU
        # Transpose here to return safetensors convention
        result = W_modified.T.to('cpu', dtype=original_dtype, non_blocking=True)

        # Cleanup
        del W_gpu, refusal_dir_gpu, refusal_normalized, projection
        del W_direction, W_direction_new, W_norm, W_modified
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return result.detach().clone()


def ablate_by_layers_sharded(
    model_name: str,
    measures: dict,
    marching_orders: list,
    output_path: str,
    norm_preserve: bool,
    projected: bool,
) -> None:
    """
    Memory-efficient ablation for sharded OR single-file safetensors models.
    Handles both local paths and HuggingFace Hub models.
    For sharded: loads one shard at a time.
    For single-file (e.g. gemma-2-2b-it): loads the model weights file once (acceptable for smaller models).
    """
    
    # Load config using transformers (handles both local and HF hub)
    print(f"Loading config for {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    
    # Determine precision (support both legacy torch_dtype and modern dtype)
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
    
    print(f"Model precision: {precision}")
    
    # Support both sharded (with index.json) and single-file (model.safetensors) models.
    # Robustly handles multimodal Gemma-4 (gemma-4-E2B-it etc.) which have audio_tower + vision_tower + language_model.
    print("Locating model weight files (sharded or single safetensors)...")
    index_path = None
    weight_map = {}
    model_dir = None
    shard_list = []
    is_sharded = False

    try:
        index_path = cached_file(model_name, "model.safetensors.index.json")
        model_dir = Path(index_path).parent
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        shard_list = sorted(set(weight_map.values()))
        is_sharded = True
        print(f"Detected sharded safetensors model ({len(shard_list)} shards)")
    except Exception:
        try:
            single_path = cached_file(model_name, "model.safetensors")
            model_dir = Path(single_path).parent
            shard_list = ["model.safetensors"]
            is_sharded = False
            print("Detected single-file safetensors model (common for small models)")
        except Exception as e2:
            raise RuntimeError(
                f"Could not locate safetensors weights for '{model_name}'. "
                "This script supports sharded models (model.safetensors.index.json + shards) "
                "or single-file model.safetensors. Pre-download the model with "
                "`huggingface-cli download <model>` if needed. "
                f"Underlying error: {e2}"
            ) from e2

    print(f"Model directory (from cache/local): {model_dir}")

    # Detect layer prefix from logical weight keys.
    # For multimodal models (gemma-4 etc) there may be audio_tower / vision_tower + language_model.
    # We must prefer the text/language_model backbone for refusal abliteration.
    def _detect_language_model_prefix(keys):
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
              "Refusal abliteration should target the text backbone.")

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

        # For sharded: skip unmodified entirely (zero-RAM copy)
        if is_sharded and shard_file not in shard_modifications:
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

            # Compute refusal direction on-the-fly (from precomputed measurements)
            refusal_dir = measures[f'refuse_{measurement}'].float()
            harmless_dir = measures[f'harmless_{layer}'].float()

            if projected:
                # Orthogonalize refusal w.r.t. harmless direction (Gram-Schmidt style)
                harmless_normalized = torch.nn.functional.normalize(harmless_dir, dim=0)
                projection_scalar = refusal_dir @ harmless_normalized
                refined = refusal_dir - projection_scalar * harmless_normalized
                refusal_dir = refined.to(precision)
                del harmless_normalized, refined

            # Optional: keep only top-k magnitude components of the direction (sparsity here means keep-fraction)
            if sparsity > 0.0:
                refusal_dir = magnitude_sparsify(refusal_dir, fraction=sparsity)

            refusal_dir = torch.nn.functional.normalize(refusal_dir, dim=-1)

            # Core weight edit
            if norm_preserve:
                state_dict[key] = modify_tensor_norm_preserved(
                    state_dict[key], refusal_dir, scale
                ).contiguous()
            else:
                state_dict[key] = modify_tensor(
                    state_dict[key], refusal_dir, scale
                ).contiguous()

            del refusal_dir, harmless_dir
            gc.collect()

        print(f"  Saving {shard_file}...")
        save_file(state_dict, f"{output_path}/{shard_file}")

        del state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Copy index only for sharded models
    print("\nCopying configuration files...")
    if is_sharded and index_path is not None:
        shutil.copy(str(index_path), f"{output_path}/model.safetensors.index.json")

    # Copy tokenizer / generation / other config files (common to both)
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
    ]

    for file in config_files:
        try:
            src_path = cached_file(model_name, file)
            if src_path and os.path.exists(src_path):
                shutil.copy(src_path, f"{output_path}/{file}")
        except Exception:
            pass

    print(f"\nModified model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient ablation (sharded or single-file safetensors) using YAML configuration."
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
        help='Preserve norms/magnitudes when ablating refusal',
    )
    parser.add_argument(
        '--projected',
        action="store_true",
        default=False,
        help='Project refusal against harmless direction and orthogonalize',
    )
    
    args = parser.parse_args()
    
    # Load YAML configuration
    with open(args.file_path, 'r') as file:
        ydata = yaml.safe_load(file)
    
    model_name = ydata.get("model")
    measurement_file = ydata.get("measurements")
    output_dir = ydata.get("output")
    ablations = ydata.get("ablate")
    
    print("=" * 60)
    print("SHARDED ABLATION CONFIGURATION")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Measurements: {measurement_file}")
    print(f"Output directory: {output_dir}")
    print(f"Number of ablations: {len(ablations)}")
    print(f"Norm preservation: {args.normpreserve}")
    print(f"Projected: {args.projected}")
    print("=" * 60)
    
    # Load measurements
    print(f"\nLoading measurements from {measurement_file}...")
    measures = torch.load(measurement_file)
    print(f"Loaded {len(measures)} measurements")
    
    # Parse ablation orders
    orders = [
        (
            int(item['layer']),
            int(item['measurement']),
            float(item['scale']),
            float(item['sparsity']),
        )
        for item in ablations
    ]
    
    print("\nAblation orders:")
    for layer, measurement, scale, sparsity in orders:
        print(f"  Layer {layer}: measurement={measurement}, scale={scale}, sparsity={sparsity}")
    
    # Perform sharded ablation
    print("\n" + "=" * 60)
    print("STARTING ABLATION")
    print("=" * 60)
    ablate_by_layers_sharded(
        model_name=model_name,
        measures=measures,
        marching_orders=orders,
        output_path=output_dir,
        norm_preserve=args.normpreserve,
        projected=args.projected,
    )
    
    print("\n" + "=" * 60)
    print("ABLATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
