from transformers import (
    TextStreamer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from argparse import ArgumentParser
import os
import torch
from utils.steering import load_steering_vectors, install_steering_hooks

parser = ArgumentParser()
parser.add_argument(
    "--model", "-m", type=str, required=True, help="Path to model directory"
)
parser.add_argument(
    "--precision",
    "-p",
    type=str,
    default="fp16",
    choices=["fp16", "bf16", "fp32"],
    help="Precision of model",
)
parser.add_argument(
    "--device",
    "-d",
    type=str,
    choices=["auto", "cuda", "cpu"],
    default="auto",
    help="Target device to process abliteration. Warning, bitsandbytes quantization DOES NOT support CPU",
)
parser.add_argument(
    "--max-new-tokens", "-n", type=int, default=256, help="Max new tokens to generate"
)
quant = parser.add_mutually_exclusive_group()
quant.add_argument(
    "--load-in-4bit",
    action="store_true",
    default=False,
    help="Load model in 4-bit precision using bitsandbytes",
)
quant.add_argument(
    "--load-in-8bit",
    action="store_true",
    default=False,
    help="Load model in 8-bit precision using bitsandbytes",
)
parser.add_argument(
    "--flash-attn", action="store_true", default=False, help="Use flash attention 2"
)
parser.add_argument(
    "--steering-vectors",
    type=str,
    default=None,
    help="Path to a steering_vectors.pt sidecar (boost mode). Defaults to "
    "<model>/steering_vectors.pt if present.",
)
parser.add_argument(
    "--steer-scale",
    type=float,
    default=1.0,
    help="Global multiplier applied over every loaded steering entry's scale",
)
args = parser.parse_args()


if __name__ == "__main__":
    if args.precision == "fp16":
        precision = torch.float16
    elif args.precision == "bf16":
        precision = torch.bfloat16
    elif args.precision == "fp32":
        precision = torch.float32

    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=precision,
            bnb_4bit_use_double_quant=True,
        )
    elif args.load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_has_fp16_weight=True,
        )
    else:
        quant_config = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=precision,
        low_cpu_mem_usage=True,
        device_map=args.device,
        quantization_config=quant_config,
        attn_implementation="flash_attention_2" if args.flash_attn else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, device_map=args.device
    )

    steering_path = args.steering_vectors or os.path.join(args.model, "steering_vectors.pt")
    if os.path.isfile(steering_path):
        steering_data = load_steering_vectors(steering_path)
        entries = steering_data["entries"]
        install_steering_hooks(model, entries, global_scale=args.steer_scale)
        print(f"! Loaded {len(entries)} steering vector(s) from {steering_path} (global scale x{args.steer_scale}):")
        for entry in entries:
            print(
                f"    layer {entry['layer']}: scale={entry['scale'] * args.steer_scale:.3f} "
                f"(ref_norm={entry.get('ref_norm', float('nan')):.2f})"
            )
    elif args.steering_vectors is not None:
        raise FileNotFoundError(f"--steering-vectors path not found: {args.steering_vectors}")

    conversation = []
    streamer = TextStreamer(tokenizer)
    print("Type /clear to clear history, /exit to quit.")
    while True:
        prompt = input("User> ")
        if prompt == "/clear":
            conversation = []
            print("! History cleared.")
            continue
        elif prompt == "/exit":
            break
        elif prompt == "":
            print("! Please type a message.")
            continue
        conversation.append({"role": "user", "content": prompt})
        toks = tokenizer.apply_chat_template(
            conversation=conversation, add_generation_prompt=True, return_tensors="pt"
        )
        if not torch.is_tensor(toks):
            # Some chat templates (e.g. gemma-4) return a BatchEncoding instead of a bare tensor
            toks = toks["input_ids"]
        gen = model.generate(
            toks.to(model.device), streamer=streamer, max_new_tokens=args.max_new_tokens
        )
        decoded = tokenizer.batch_decode(
            gen[0][len(toks[0]) :], skip_special_tokens=True
        )
        conversation.append({"role": "assistant", "content": "".join(decoded)})
