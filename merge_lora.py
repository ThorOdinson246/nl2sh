#!/usr/bin/env python3
"""Merge a trained LoRA adapter into its base model, in bf16 (never fp16 --
a fp16 merge of a bf16-trained adapter silently loses precision on outlier
weights, per docs/GENERATION_FEASIBILITY.md's export-path notes).

Copies tokenizer files AND the chat template into the merged output dir --
a GGUF converted without the chat template gets prompted wrongly by every
downstream tool, which is easy to misdiagnose as a bad fine-tune rather than
a missing template.
"""
import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base model name or path")
    ap.add_argument("--adapter", required=True, help="dir containing the trained LoRA adapter")
    ap.add_argument("--out", required=True, help="output dir for the merged model")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading base model {args.base} in bf16")
    base_model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)

    print(f"loading adapter from {args.adapter}")
    peft_model = PeftModel.from_pretrained(base_model, args.adapter)

    print("merging (bf16, not fp16 -- see module docstring)")
    merged = peft_model.merge_and_unload()

    print(f"saving merged model to {out_dir}")
    merged.save_pretrained(out_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    tokenizer.save_pretrained(out_dir)

    # Belt-and-suspenders: explicitly verify the chat template made it across,
    # since a silently-missing template is the specific failure mode this
    # module's docstring warns about.
    chat_template_path = out_dir / "chat_template.jinja"
    tokenizer_config_path = out_dir / "tokenizer_config.json"
    has_template = chat_template_path.exists() or (
        tokenizer_config_path.exists()
        and "chat_template" in tokenizer_config_path.read_text()
    )
    if not has_template:
        raise RuntimeError(
            f"No chat template found in {out_dir} after save_pretrained. "
            "A GGUF built from this merge will be prompted incorrectly by "
            "every downstream tool. Check the adapter dir's tokenizer_config.json."
        )
    print("chat template present in merged output -- OK")
    print(f"DONE -> {out_dir}")


if __name__ == "__main__":
    main()
