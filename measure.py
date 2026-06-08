import gc
import torch
from argparse import ArgumentParser
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoModelForImageTextToText
from transformers import AutoTokenizer
from transformers import AutoProcessor
from transformers import BitsAndBytesConfig
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from utils.data import load_data
from utils.models import has_tied_weights
from utils.clip import magnitude_clip


def welford_gpu_batched_multilayer_float32(
    formatted_prompts: list[str],
    desc: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    layer_indices: list[int],
    pos: int = -1,
    batch_size: int = 1,
    clip: float = 1.0,
    processor = None,  # Add processor parameter
    is_vision_model: bool = False,  # Add flag for vision models
) -> dict[int, torch.Tensor]:
    text_config = model.config
    if hasattr(text_config, "text_config"):
        text_config = text_config.text_config
    vocab_size = text_config.vocab_size

    max_tokens = 1

    means = {layer_idx: None for layer_idx in layer_indices}
    counts = {layer_idx: 0 for layer_idx in layer_indices}
    dtype = model.dtype

    for i in tqdm(range(0, len(formatted_prompts), batch_size), desc=desc):
        batch_prompts = formatted_prompts[i:i+batch_size]

        if is_vision_model and processor is not None:
            # For vision models, use the processor with text-only input
            batch_encoding = processor(
                text=batch_prompts,
                return_tensors="pt",
                padding=True,
            )
        else:
            # For text-only models, use the tokenizer
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'

            batch_encoding = tokenizer(
                batch_prompts,
                padding=True,
                padding_side='left',
                return_tensors="pt",
            )
        
        batch_input = batch_encoding['input_ids'].to(model.device)
        batch_mask = batch_encoding['attention_mask'].to(model.device)

        # Use generate to get hidden states at the first generated token position
        raw_output = model.generate(
            batch_input,
            attention_mask=batch_mask,
            max_new_tokens=max_tokens,
            return_dict_in_generate=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        del batch_input, batch_mask
        hidden_states = raw_output.hidden_states[0]  # First generation step
        del raw_output

        # Process layers with Welford in float32
        for layer_idx in layer_indices:
            # Cast to float32 for accumulation
            current_hidden = hidden_states[layer_idx][:, pos, :].float()
            if (clip < 1.0):
                current_hidden = magnitude_clip(current_hidden, clip)

            batch_size_actual = current_hidden.size(dim=0)
            total_count = counts[layer_idx] + batch_size_actual

            if means[layer_idx] is None:
                # Initialize mean in float32
                means[layer_idx] = current_hidden.mean(dim=0)
            else:
                # All operations in float32 (means[layer_idx] is already float32)
                delta = current_hidden - means[layer_idx]
                means[layer_idx] += delta.sum(dim=0) / total_count

            counts[layer_idx] = total_count
            del current_hidden

        del hidden_states
        torch.cuda.empty_cache()

    # Cast back to model dtype and move to CPU
    return_dict = {
        layer_idx: mean.to(device="cpu") 
        for layer_idx, mean in means.items()
    }
    del means
    torch.cuda.empty_cache()
    return return_dict

def format_chats(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    prompt_list: list[str],
    processor = None,
):
    # Use processor's tokenizer if available, otherwise use tokenizer directly
    actual_tokenizer = processor.tokenizer if processor is not None else tokenizer
    
    result_formatted = [
        actual_tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": inst}],
            add_generation_prompt=True,
            add_special_tokens=False,
            tokenize=False,
        )
        for inst in prompt_list
    ]
    return result_formatted

# ============================================================
# Static conversation support (good/bad JSONL with full messages)
# ============================================================

def load_conversations(jsonl_path: str):
    """Stream conversations from a JSONL file. Each line = list of messages."""
    import json
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                messages = json.loads(line)
                if isinstance(messages, list) and len(messages) > 0:
                    yield messages
            except Exception:
                continue


def get_assistant_end_positions(tokenizer, messages: list[dict]) -> list[int]:
    """
    Return the token indices (in the full conversation) of the last token
    of each assistant reply. We build the template incrementally for accuracy.
    """
    positions = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            convo_so_far = messages[: i + 1]
            input_ids = tokenizer.apply_chat_template(
                convo_so_far,
                tokenize=True,
                add_generation_prompt=False,
            )
            if isinstance(input_ids, list) and len(input_ids) > 0:
                positions.append(len(input_ids) - 1)
    return positions


def compute_means_from_conversations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    conversations: list[list[dict]],
    desc: str,
    dialogue_batch_size: int = 4,
    max_length: int | None = None,
    clip: float = 1.0,
) -> dict[int, torch.Tensor]:
    """
    Memory-efficient extraction of per-layer means from full conversations.
    - Processes conversations in small groups (dialogue_batch_size).
    - Drops any conversation whose tokenized length > max_length.
    - Only updates running Welford means at the last token of each assistant turn.
    - Aggressively clears memory after each micro-batch.
    """
    from utils.clip import magnitude_clip

    layer_base = model.model
    if hasattr(layer_base, "language_model"):
        layer_base = layer_base.language_model
    num_layers = len(layer_base.layers)

    means = {layer_idx: None for layer_idx in range(num_layers)}
    counts = {layer_idx: 0 for layer_idx in range(num_layers)}

    for start_idx in tqdm(range(0, len(conversations), dialogue_batch_size), desc=desc):
        batch = conversations[start_idx : start_idx + dialogue_batch_size]

        prepared = []  # (input_ids_list, assistant_positions)
        for convo in batch:
            try:
                input_ids = tokenizer.apply_chat_template(
                    convo, tokenize=True, add_generation_prompt=False
                )
                if not isinstance(input_ids, list):
                    continue
                if max_length is not None and len(input_ids) > max_length:
                    continue  # drop entire conversation (user requirement)
                positions = get_assistant_end_positions(tokenizer, convo)
                if positions:
                    prepared.append((input_ids, positions))
            except Exception:
                continue

        if not prepared:
            continue

        # Left-pad the batch for causal LM forward
        max_len = max(len(ids) for ids, _ in prepared)
        batch_input = []
        batch_mask = []
        adjusted_pos_lists = []

        for ids, positions in prepared:
            pad_len = max_len - len(ids)
            padded = [tokenizer.pad_token_id] * pad_len + ids
            mask = [0] * pad_len + [1] * len(ids)
            batch_input.append(padded)
            batch_mask.append(mask)
            adjusted_pos_lists.append([p + pad_len for p in positions])

        input_tensor = torch.tensor(batch_input, dtype=torch.long, device=model.device)
        mask_tensor = torch.tensor(batch_mask, dtype=torch.long, device=model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_tensor,
                attention_mask=mask_tensor,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states  # (embed, layer0, layer1, ..., layerN)

        # Update means only at the desired positions
        for sidx, pos_list in enumerate(adjusted_pos_lists):
            for pos in pos_list:
                for layer_idx in range(num_layers):
                    # hidden_states[0] = after embedding
                    # hidden_states[layer_idx + 1] = after transformer layer layer_idx
                    hs = hidden_states[layer_idx + 1] if len(hidden_states) > num_layers else hidden_states[layer_idx]
                    vec = hs[sidx, pos, :].float()

                    if clip < 1.0:
                        vec = magnitude_clip(vec, clip)

                    total = counts[layer_idx] + 1
                    if means[layer_idx] is None:
                        means[layer_idx] = vec
                    else:
                        delta = vec - means[layer_idx]
                        means[layer_idx] += delta / total
                    counts[layer_idx] = total

        # Aggressive cleanup
        del outputs, hidden_states, input_tensor, mask_tensor
        torch.cuda.empty_cache()

    # Return on CPU
    return {
        layer_idx: mean.to("cpu")
        for layer_idx, mean in means.items()
        if mean is not None
    }


def compute_refusals(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    harmful_list: list[str],
    harmless_list: list[str],
    projected: bool = False,
    inference_batch_size: int = 32,
    clip: float = 1.0,
    processor = None,  # Add processor parameter
    is_vision_model: bool = False,  # Add flag for vision models
) -> torch.Tensor:
    # dtype = model.dtype
    layer_base = model.model
    if hasattr(layer_base,"language_model"):
        layer_base = layer_base.language_model
    num_layers = len(layer_base.layers)
    pos = -1
    # option for layer sweep
    focus_layers = range(num_layers)

    harmful_formatted = format_chats(tokenizer=tokenizer, prompt_list=harmful_list, processor=processor)
    harmful_means = welford_gpu_batched_multilayer_float32(
        harmful_formatted, "Generating harmful outputs", model, tokenizer, 
        focus_layers, pos, inference_batch_size, clip, processor, is_vision_model
    )
    torch.cuda.empty_cache()
    del harmful_formatted
    harmless_formatted = format_chats(tokenizer=tokenizer, prompt_list=harmless_list, processor=processor)
    harmless_means = welford_gpu_batched_multilayer_float32(
        harmless_formatted, "Generating harmless outputs", model, tokenizer, 
        focus_layers, pos, inference_batch_size, clip, processor, is_vision_model
    )
    del harmless_formatted

    results = {}
    results["layers"] = num_layers

    # Keep all results in 32-bit float for analysis/ablation
    for layer in tqdm(focus_layers,desc="Compiling layer measurements"):
        harmful_mean = harmful_means[layer]
        results[f'harmful_{layer}'] = harmful_mean
        harmless_mean = harmless_means[layer]
        results[f'harmless_{layer}'] = harmless_mean
        refusal_dir = harmful_mean - harmless_mean

        if projected:
            # Compute Gram-Schmidt second orthogonal vector/direction to remove harmless direction interference from refusal direction
            # Normalize harmless_mean to avoid numerical issues in projection calculation
            harmless_normalized = torch.nn.functional.normalize(harmless_mean.float(), dim=0)

            # Project and subtract contribution along harmless direction
            projection_scalar = refusal_dir @ harmless_normalized

            # Resulting refusal direction should minimize impact along harmless direction
            refusal_dir = refusal_dir - projection_scalar * harmless_normalized
        # otherwise default to stock abliteration refusal direction calculation

        results[f'refuse_{layer}'] = refusal_dir

    torch.cuda.empty_cache()
    gc.collect()
    return results


def compute_refusal_directions_from_static_means(
    harmful_means: dict,
    harmless_means: dict,
    projected: bool = False,
) -> dict:
    """
    Pure static/offline version of refusal direction computation.
    Does NOT require a model or any inference. Only vector math on pre-computed means.

    This is the core of the new "static data training" path.
    You can pre-extract means once (expensive), then cheaply experiment with different
    projection settings, different layer combinations, etc.
    """
    results = {}
    # Try to recover num_layers
    if "layers" in harmful_means:
        num_layers = harmful_means["layers"]
    else:
        # Infer from keys like harmful_0, harmful_1, ...
        num_layers = max(
            (int(k.split("_", 1)[1]) for k in harmful_means.keys() if k.startswith("harmful_")),
            default=0
        ) + 1

    results["layers"] = num_layers
    results["source"] = "static"

    focus_layers = range(num_layers)

    for layer in tqdm(focus_layers, desc="Computing refusal directions from static means"):
        h_key = f"harmful_{layer}"
        hl_key = f"harmless_{layer}"

        if h_key not in harmful_means or hl_key not in harmless_means:
            print(f"Warning: missing means for layer {layer}, skipping.")
            continue

        harmful_mean = harmful_means[h_key]
        harmless_mean = harmless_means[hl_key]

        # keep as float32 for numerical stability
        refusal_dir = (harmful_mean.float() - harmless_mean.float())

        if projected:
            harmless_normalized = torch.nn.functional.normalize(harmless_mean.float(), dim=0)
            projection_scalar = refusal_dir @ harmless_normalized
            refusal_dir = refusal_dir - projection_scalar * harmless_normalized

        results[f"harmful_{layer}"] = harmful_mean
        results[f"harmless_{layer}"] = harmless_mean
        results[f"refuse_{layer}"] = refusal_dir

    return results

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Measure models for analysis and abliteration. "
                    "Supports both live inference (default) and static/offline mode using pre-computed activations "
                    "(see --static-harmful-means / --static-harmless-means on the feature/static-activations branch)."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        # Not always required: when using --static-*-means we can work purely from pre-computed activations
        required=False,
        help="Local model directory or HuggingFace model ID (not needed in --static-* mode)",
    )
    parser.add_argument(
        "--quant-measure", "-q",
        type=str,
        choices=["4bit", "8bit"],
        default=None,
        help="Perform measurement using 4bit or 8bit bitsandbytes quant"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size during inference/calibration; default 32, stick to powers of 2 (higher will use more VRAM)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        required=True,
        help="Output file for measurements"
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=1.0,
        help="Fraction of prompt activation to clip by magnitude",
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        default=False,
        help="Use Flash Attention 2"
    )
    parser.add_argument(
        "--data-harmful",
        type=str,
        default=None,
        help="Harmful prompts file"
    )
    parser.add_argument(
        "--data-harmless",
        type=str,
        default=None,
        help="Harmless prompts file"
    )
    parser.add_argument(
        "--deccp",
        action="store_true",
        default=False,
        help="For Chinese models, add topics to harmful prompts",
    )
    parser.add_argument(
        "--projected",
        action="store_true",
        default=False,
        help="Remove projection along harmless direction from refusal direction",
    )

    # --- Static / offline activation support (new feature on this branch) ---
    parser.add_argument(
        "--static-harmful-means",
        type=str,
        default=None,
        help="Path to a previous .refuse / .pt file containing pre-computed 'harmful_*' layer means. "
             "When provided together with --static-harmless-means, no model will be loaded and no inference is performed.",
    )
    parser.add_argument(
        "--static-harmless-means",
        type=str,
        default=None,
        help="Path to a previous .refuse / .pt file containing pre-computed 'harmless_*' layer means.",
    )

    # New static conversation mode using full message JSONL (good vs bad behavior)
    parser.add_argument(
        "--good-jsonl",
        type=str,
        default=None,
        help="Path to JSONL file containing good (desired) conversations in messages format. "
             "Each line should be a JSON array of {'role': ..., 'content': ...}. "
             "Used as the 'harmless' side in static mode.",
    )
    parser.add_argument(
        "--bad-jsonl",
        type=str,
        default=None,
        help="Path to JSONL file containing bad (refusing / undesired) conversations in messages format. "
             "Each line should be a JSON array of {'role': ..., 'content': ...}. "
             "Used as the 'harmful' side in static mode.",
    )

    args = parser.parse_args()

    # In pure static-means mode we don't need the model arg
    using_static_means = bool(args.static_harmful_means and args.static_harmless_means)
    using_static_conversations = bool(args.good_jsonl and args.bad_jsonl)

    if not (using_static_means or using_static_conversations):
        assert isinstance(args.model, str), "--model is required for live or conversation-static mode"
    assert isinstance(args.output, str), "--output is always required"

    # ============================================================
    # Static / offline paths (feature/static-activations branch)
    # ============================================================
    if using_static_conversations:
        print("=== STATIC CONVERSATIONS MODE (good-jsonl vs bad-jsonl, forward only) ===")
        print(f"Good (desired) conversations: {args.good_jsonl}")
        print(f"Bad (refusing) conversations : {args.bad_jsonl}")

        # We still need the model to run forward on the dialogues
        # (but we will load it later in the normal flow or here)
        # For now we let the normal model loading happen below, then branch.

    elif using_static_means:
        print("=== STATIC MEANS MODE (no model, no inference) ===")
        print(f"Loading pre-computed harmful means from: {args.static_harmful_means}")
        print(f"Loading pre-computed harmless means from: {args.static_harmless_means}")

        harmful_means = torch.load(args.static_harmful_means, map_location="cpu")
        harmless_means = torch.load(args.static_harmless_means, map_location="cpu")

        results = compute_refusal_directions_from_static_means(
            harmful_means, harmless_means, args.projected
        )

        print(f"Saving refusal information (static) to {args.output}...")
        torch.save(results, args.output)
        print("Done.")
        import sys
        sys.exit(0)

    # --- Live inference path continues below ---
    torch.inference_mode()
    torch.set_grad_enabled(False)

    model = args.model
    model_config = AutoConfig.from_pretrained(model)
    model_type = getattr(model_config,"model_type")

    # Get the precision/dtype from config, with proper fallback (prefer modern "dtype")
    if hasattr(model_config, "dtype") and model_config.dtype is not None:
        precision = model_config.dtype
    elif hasattr(model_config, "torch_dtype") and model_config.torch_dtype is not None:
        precision = model_config.torch_dtype
    else:
        # Fallback to bfloat16 if available, otherwise float16
        precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Convert string dtype to torch dtype if needed
    if isinstance(precision, str):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        precision = dtype_map.get(precision, torch.bfloat16)

    has_vision = False
    if hasattr(model_config, "vision_config") and getattr(model_config, "vision_config", None):
        has_vision = True
    # Also detect multimodal conditional generation models (e.g. Gemma4ForConditionalGeneration, gemma-4 series)
    archs = getattr(model_config, "architectures", []) or []
    model_type = getattr(model_config, "model_type", "") or ""
    if any("ConditionalGeneration" in str(a) for a in archs) or "gemma4" in str(model_type).lower():
        has_vision = True
    model_loader = AutoModelForCausalLM
    if has_vision:
        model_loader = AutoModelForImageTextToText

    quant_config = None
    qbit = args.quant_measure
    # autodetect BitsAndBytes quant; overrides option
    if hasattr(model_config,"quantization_config"):
        bnb_config = getattr(model_config, "quantization_config")
        if (bnb_config["load_in_4bit"] == True):
            qbit = "4bit"
            # Override precision with compute dtype from quant config if available
            if "bnb_4bit_compute_dtype" in bnb_config and bnb_config["bnb_4bit_compute_dtype"]:
                compute_dtype = bnb_config["bnb_4bit_compute_dtype"]
                if isinstance(compute_dtype, str):
                    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
                    precision = dtype_map.get(compute_dtype, precision)
                else:
                    precision = compute_dtype
                print(f"Using compute dtype from quant config: {precision}")
        elif (bnb_config["load_in_8bit"] == True):
            qbit = "8bit"

    if qbit == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=precision,
            bnb_4bit_use_double_quant=True,
        )
    elif qbit == "8bit":
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
#            llm_int8_enable_fp32_cpu_offload=True,
#            llm_int8_has_fp16_weight=True,
        )    

    if isinstance(args.data_harmful, str):
        harmful_list = load_data(args.data_harmful)
    else:
        harmful_list = load_data("./data/harmful.parquet")
    if isinstance(args.data_harmless, str):
        harmless_list = load_data(args.data_harmless)
    else:
        harmless_list = load_data("./data/harmless.parquet")

    if args.deccp:
        deccp_list = load_dataset("augmxnt/deccp", split="censored")
        harmful_list += deccp_list["text"]

    # Assume "cuda" device for now; refactor later if there's demand for other GPU-accelerated platforms
    if hasattr(model_config, "quantization_config"):
        # Use the chosen loader (may be VLM for gemma-4 etc.) even for pre-quantized models
        model = model_loader.from_pretrained(
            args.model,
#            trust_remote_code=True,
            dtype=precision,
            device_map="auto",
            attn_implementation="flash_attention_2" if args.flash_attn else None,
        )
    else:
        model = model_loader.from_pretrained(
            args.model,
#            trust_remote_code=True,
            dtype=precision,
            low_cpu_mem_usage=True,
            device_map="auto",
            quantization_config=quant_config,
            attn_implementation="flash_attention_2" if args.flash_attn else None,
        )
    model.requires_grad_(False)
    if has_tied_weights(model_type):
        model.tie_weights()

    # point to base of language model
    layer_base = model.model
    if hasattr(layer_base,"language_model"):
        layer_base = layer_base.language_model

    # Load processor for vision models, tokenizer for text-only models
    processor = None
    if has_vision:
        try:
            processor = AutoProcessor.from_pretrained(
                args.model,
                device_map="cuda",
                padding=True,
            )
            tokenizer = processor.tokenizer
            print("Loaded processor for vision model")
        except (IndexError, Exception) as e:
            # If processor loading fails, fall back to tokenizer only
            print(f"Could not load processor ({e}), falling back to tokenizer only")
            has_vision = False
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
#                trust_remote_code=True,
                device_map="cuda",
                padding=True,
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
#            trust_remote_code=True,
            device_map="cuda",
            padding=True,
        )

    # --- Static conversations path (good/bad full dialogues) ---
    if args.good_jsonl and args.bad_jsonl:
        print("Loading conversations from JSONL (streaming)...")
        good_convos = list(load_conversations(args.good_jsonl))
        bad_convos = list(load_conversations(args.bad_jsonl))
        print(f"Loaded {len(good_convos)} good conversations, {len(bad_convos)} bad conversations.")

        # Ensure we can pad (needed when dialogue_batch_size > 1)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

        # In this mode:
        #   bad  = harmful side (undesired / refusing behavior)
        #   good = harmless side (desired behavior)
        harmful_means = compute_means_from_conversations(
            model, tokenizer, bad_convos,
            desc="Extracting means from BAD conversations",
            dialogue_batch_size=args.batch_size,
            max_length=getattr(model.config, "max_position_embeddings", None),
            clip=args.clip,
        )
        torch.cuda.empty_cache()

        harmless_means = compute_means_from_conversations(
            model, tokenizer, good_convos,
            desc="Extracting means from GOOD conversations",
            dialogue_batch_size=args.batch_size,
            max_length=getattr(model.config, "max_position_embeddings", None),
            clip=args.clip,
        )
        torch.cuda.empty_cache()

        results = compute_refusal_directions_from_static_means(
            harmful_means, harmless_means, args.projected
        )
        results["source"] = "static-conversations"
        results["good_jsonl"] = args.good_jsonl
        results["bad_jsonl"] = args.bad_jsonl

        print(f"Saving refusal information to {args.output}...")
        torch.save(results, args.output)
        import sys
        sys.exit(0)

    print("Computing refusal information...")
    results = {}
    results = compute_refusals(
        model, tokenizer, harmful_list, harmless_list,
        args.projected, args.batch_size, args.clip, processor, has_vision
    )

    print(f"Saving refusal information to {args.output}...")
    torch.save(results, args.output)
