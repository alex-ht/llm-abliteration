import torch
from transformers import PreTrainedModel


def load_steering_vectors(path: str) -> dict:
    """Load a steering_vectors.pt sidecar produced by sharded_ablate.py's boost mode."""
    return torch.load(path, map_location="cpu")


def get_decoder_layers(model: PreTrainedModel) -> torch.nn.ModuleList:
    """
    Locate the decoder layer list for a (possibly multimodal) causal LM.
    Mirrors the base/.language_model fallback used in measure.py and sharded_ablate.py.
    """
    base = getattr(model, "model", model)
    base = getattr(base, "language_model", base)
    if not hasattr(base, "layers"):
        raise ValueError(
            "Could not locate decoder layers on this model (looked for "
            "model.layers / model.language_model.layers)."
        )
    return base.layers


def _make_hook(vector: torch.Tensor, scale: float):
    def _hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            delta = (scale * vector).to(device=hidden_states.device, dtype=hidden_states.dtype)
            return (hidden_states + delta,) + output[1:]
        delta = (scale * vector).to(device=output.device, dtype=output.dtype)
        return output + delta
    return _hook


def install_steering_hooks(
    model: PreTrainedModel, entries: list[dict], global_scale: float = 1.0
) -> list[torch.utils.hooks.RemovableHandle]:
    """
    Register a forward hook per steering entry on the model's decoder layers. Each hook
    adds `entry["scale"] * global_scale * entry["vector"]` to that layer's output hidden
    states on every forward call (prefill and incremental decode alike -- broadcasting
    over [batch, seq, hidden] needs no special-casing between the two).
    """
    layers = get_decoder_layers(model)
    handles = []
    for entry in entries:
        layer_idx = entry["layer"]
        if not (0 <= layer_idx < len(layers)):
            raise ValueError(
                f"Steering entry targets layer {layer_idx}, but this model only has "
                f"{len(layers)} decoder layers."
            )
        hook = _make_hook(entry["vector"], entry["scale"] * global_scale)
        handles.append(layers[layer_idx].register_forward_hook(hook))
    return handles


def remove_steering_hooks(handles: list) -> None:
    for handle in handles:
        handle.remove()
